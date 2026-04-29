"""Algorithm 1 — CANVAS sequential generation with memory + diskcache resume."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from diskcache import Cache
from pydantic import TypeAdapter

from .agents import (
    ImagePipeline,
    crop_to_anchor,
    extract_location_anchor,
    extract_visibility,
    generate,
    plan,
    retrieve,
    segment_to_anchor,
    select,
)
from .config import CACHE_TAG_GENERATE, Settings
from .memory import Memory, safe_filename
from .schemas import (
    AnchorSet,
    BackgroundPlan,
    BBox,
    CandidateScore,
    CharacterState,
    FrameVisibility,
    Plan,
    PropState,
    Shot,
    ShotResultSummary,
    Story,
    make_generation_key,
)


@dataclass
class ShotResult:
    shot: Shot
    anchors: AnchorSet
    candidates: list[Path]
    scores: list[CandidateScore]
    chosen: Path


def _load_image_pipeline(settings: Settings) -> ImagePipeline:
    import torch
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(
        settings.image_model, torch_dtype=torch.bfloat16
    ).to(settings.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _seed_canonical_anchors(story: Story, memory: Memory) -> None:
    for c in story.characters:
        if not c.reference_image:
            continue
        try:
            memory.add_canonical(c.id, CharacterState.default, Path(c.reference_image))
        except FileNotFoundError:
            continue


def _cached_generate(
    cache: Cache, shot: Shot, anchors: AnchorSet, bg_plan: BackgroundPlan | None,
    settings: Settings, pipeline: ImagePipeline, seed: int,
) -> list[Path]:
    key = make_generation_key(shot, anchors, seed, settings.k_candidates)
    hit = cache.get(key)
    if hit and all(Path(p).exists() for p in hit):
        return [Path(p) for p in hit]
    paths = generate(shot, anchors, bg_plan, settings, pipeline, seed=seed)
    cache.set(key, [str(p) for p in paths], tag=CACHE_TAG_GENERATE)
    return paths


_RESULTS_ADAPTER = TypeAdapter(list[ShotResultSummary])


def _write_results(out_dir: Path, results: list[ShotResult]) -> None:
    summary = [
        ShotResultSummary(shot_index=r.shot.index, chosen=str(r.chosen), scores=r.scores)
        for r in results
    ]
    (out_dir / "results.json").write_bytes(_RESULTS_ADAPTER.dump_json(summary, indent=2))


def _subject_anchor(chosen: Path, bbox, dest: Path, settings: Settings) -> Path:
    """Crop a subject anchor; segment + neutral-fill when enabled."""
    if settings.enable_segmentation:
        return segment_to_anchor(
            chosen, bbox, dest,
            model_name=settings.segment_model, bg_color=settings.segment_bg_color,
        )
    return crop_to_anchor(chosen, bbox, dest)


def _refresh_subject_anchors(
    chosen: Path,
    visibility_items: list[tuple[str, bool, BBox | None]],
    expected_states: dict[str, str],
    skip_states: set[str], add_anchor: Callable[[str, str, Path], Path],
    file_prefix: str, crop_dir: Path, settings: Settings,
) -> None:
    """Refresh memory anchors for chars or props (same shape, differs only by add_anchor)."""
    by_id = {v[0]: v for v in visibility_items}  # id -> (id, visible, bbox)
    for entity_id, state in expected_states.items():
        if state in skip_states:
            continue
        item = by_id.get(entity_id)
        if visibility_items and (item is None or not item[1]):
            continue
        bbox = item[2] if item else None
        if bbox:
            fname = f"{file_prefix}__{safe_filename(entity_id)}__{safe_filename(state)}.png"
            anchor = _subject_anchor(chosen, bbox, crop_dir / fname, settings)
        else:
            anchor = chosen
        add_anchor(entity_id, state, anchor)


def _refresh_anchors(
    shot: Shot, chosen: Path, memory: Memory, visibility: FrameVisibility,
    crop_dir: Path, settings: Settings,
) -> None:
    """Algorithm 4 — VLM gates anchor refresh; subject-only anchors via bbox + segmentation."""
    crop_dir.mkdir(parents=True, exist_ok=True)

    _refresh_subject_anchors(
        chosen,
        [(cv.character_id, cv.visible, cv.bbox) for cv in visibility.characters],
        shot.character_states, {CharacterState.not_present},
        memory.add_character, "char", crop_dir, settings,
    )
    _refresh_subject_anchors(
        chosen,
        [(pv.prop_id, pv.visible, pv.bbox) for pv in visibility.props],
        shot.prop_states, {PropState.not_visible, PropState.not_present},
        memory.add_prop, "prop", crop_dir, settings,
    )

    if shot.location_id and visibility.location_visible:
        if settings.enable_segmentation:
            subject_bboxes = [
                cv.bbox for cv in visibility.characters if cv.visible and cv.bbox
            ] + [pv.bbox for pv in visibility.props if pv.visible and pv.bbox]
            anchor = extract_location_anchor(
                chosen, subject_bboxes,
                crop_dir / f"loc__{safe_filename(shot.location_id)}.png",
                model_name=settings.segment_model, bg_color=settings.segment_bg_color,
                enable_inpainting=settings.enable_inpainting,
            )
        else:
            anchor = chosen
        memory.add_location(shot.location_id, anchor)


async def run(
    story: Story, settings: Settings,
    image_pipeline: ImagePipeline | None = None,
    seed_maker: Callable[[Shot], int] | None = None,
) -> tuple[Plan, list[ShotResult]]:
    settings.ensure_dirs()
    memory = Memory.load_or_empty(settings.out_dir / "memory")
    _seed_canonical_anchors(story, memory)
    pipe = image_pipeline if image_pipeline is not None else _load_image_pipeline(settings)
    seed_of = seed_maker or (lambda s: settings.seed + s.index * settings.seed_stride)

    p = await plan(story, settings)
    (settings.out_dir / "plan.json").write_text(p.model_dump_json(indent=2))
    bg_by_shot = {bp.shot_index: bp for bp in p.background_plans}

    results: list[ShotResult] = []
    with Cache(str(settings.cache_dir), tag_index=True) as cache:
        for shot in p.shots:
            anchors = retrieve(shot, p, memory)
            candidates = _cached_generate(
                cache, shot, anchors, bg_by_shot.get(shot.index),
                settings, pipe, seed=seed_of(shot),
            )
            chosen, scores = await select(candidates, shot, memory, settings)

            memory.add_frame(shot.index, chosen)
            visibility = await extract_visibility(shot, chosen, settings)
            _refresh_anchors(
                shot, chosen, memory, visibility,
                crop_dir=settings.out_dir / "crops" / f"shot_{shot.index:04d}",
                settings=settings,
            )
            memory.save()

            results.append(ShotResult(
                shot=shot, anchors=anchors, candidates=candidates,
                scores=scores, chosen=chosen,
            ))
            _write_results(settings.out_dir, results)

    return p, results


def run_sync(
    story: Story, settings: Settings,
    image_pipeline: ImagePipeline | None = None,
    seed_maker: Callable[[Shot], int] | None = None,
) -> tuple[Plan, list[ShotResult]]:
    """Synchronous wrapper for `run`."""
    return asyncio.run(run(story, settings, image_pipeline, seed_maker))
