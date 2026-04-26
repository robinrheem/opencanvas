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
    extract_visibility_async,
    generate,
    plan_async,
    retrieve,
    segment_to_anchor,
    select_async,
)
from .config import CACHE_TAG_GENERATE, Settings
from .memory import Memory
from .schemas import (
    AnchorSet,
    BackgroundPlan,
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


def _load_image_pipeline(settings: Settings) -> ImagePipeline:
    import torch
    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        settings.image_model, torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


@dataclass
class ShotResult:
    shot: Shot
    anchors: AnchorSet
    candidates: list[Path]
    scores: list[CandidateScore]
    chosen: Path


def _seed_canonical_anchors(story: Story, memory: Memory) -> None:
    for c in story.characters:
        if not c.reference_image:
            continue
        try:
            memory.set_character_canonical(c.id, "default", Path(c.reference_image))
        except FileNotFoundError:
            continue


def _cached_generate(
    cache: Cache,
    shot: Shot,
    anchors: AnchorSet,
    background_plan: BackgroundPlan | None,
    settings: Settings,
    pipeline: ImagePipeline,
    seed: int,
) -> list[Path]:
    key = make_generation_key(shot, anchors, seed, settings.k_candidates)
    hit = cache.get(key)
    if hit and all(Path(p).exists() for p in hit):
        return [Path(p) for p in hit]
    paths = generate(shot, anchors, background_plan, settings, pipeline, seed=seed)
    cache.set(key, [str(p) for p in paths], tag=CACHE_TAG_GENERATE)
    return paths


_RESULTS_ADAPTER = TypeAdapter(list[ShotResultSummary])


def _write_results(out_dir: Path, results: list[ShotResult]) -> None:
    summary = [
        ShotResultSummary(shot_index=r.shot.index, chosen=str(r.chosen), scores=r.scores)
        for r in results
    ]
    (out_dir / "results.json").write_bytes(_RESULTS_ADAPTER.dump_json(summary, indent=2))


async def run_async(
    story: Story,
    settings: Settings,
    image_pipeline: ImagePipeline | None = None,
    seed_maker: Callable[[Shot], int] | None = None,
) -> tuple[Plan, list[ShotResult]]:
    settings.ensure_dirs()
    memory = Memory.load_or_empty(settings.out_dir / "memory")
    _seed_canonical_anchors(story, memory)
    pipe = image_pipeline if image_pipeline is not None else _load_image_pipeline(settings)
    seed_of = seed_maker or (lambda s: settings.seed + s.index * settings.seed_stride)

    p = await plan_async(story, settings)
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
            chosen, scores = await select_async(candidates, shot, memory, settings)

            memory.set_frame(shot.index, chosen)
            visibility = await extract_visibility_async(shot, chosen, settings)
            _update_anchors_from_frame(
                shot, chosen, memory, visibility,
                crop_dir=settings.out_dir / "crops" / f"shot_{shot.index:04d}",
                settings=settings,
            )
            memory.save()

            results.append(
                ShotResult(
                    shot=shot, anchors=anchors, candidates=candidates,
                    scores=scores, chosen=chosen,
                )
            )
            _write_results(settings.out_dir, results)

    return p, results


def run(
    story: Story,
    settings: Settings,
    image_pipeline: ImagePipeline | None = None,
    seed_maker: Callable[[Shot], int] | None = None,
) -> tuple[Plan, list[ShotResult]]:
    """Synchronous wrapper for `run_async`."""
    return asyncio.run(run_async(story, settings, image_pipeline, seed_maker))


def _extract_anchor(
    chosen: Path, bbox, dest: Path, settings: Settings
) -> Path:
    """Crop subject from chosen frame; optionally segment + composite on neutral bg
    to suppress background drift in downstream multi-ref generation."""
    if settings.enable_segmentation:
        return segment_to_anchor(
            chosen, bbox, dest,
            model_name=settings.segment_model,
            bg_color=settings.segment_bg_color,
        )
    return crop_to_anchor(chosen, bbox, dest)


def _update_anchors_from_frame(
    shot: Shot,
    chosen: Path,
    memory: Memory,
    visibility: FrameVisibility,
    crop_dir: Path,
    settings: Settings,
) -> None:
    """Algorithm 4 — VLM gates anchor refresh; subject-only anchors via bbox + segmentation.

    Characters and props with a bbox are extracted from the chosen frame
    (Tables 27, 29). When `settings.enable_segmentation` is True, the bbox crop
    is further bg-removed and composited onto a neutral mid-gray plate so the
    anchor carries identity but not the chosen frame's environment. Locations
    use the full frame (Table 28 prescribes no crop).
    """
    crop_dir.mkdir(parents=True, exist_ok=True)
    char_visibility = {cv.character_id: cv for cv in visibility.characters}
    prop_visibility = {pv.prop_id: pv for pv in visibility.props}

    for cid, state in shot.character_states.items():
        if state == CharacterState.not_present:
            continue
        cv = char_visibility.get(cid)
        if visibility.characters and (cv is None or not cv.visible):
            continue
        if cv and cv.bbox:
            anchor = _extract_anchor(
                chosen, cv.bbox, crop_dir / f"char__{cid}__{state}.png", settings
            )
        else:
            anchor = chosen
        memory.set_character(cid, state, anchor)

    if shot.location_id and visibility.location_visible:
        if settings.enable_segmentation:
            subject_bboxes = [
                cv.bbox for cv in visibility.characters if cv.visible and cv.bbox
            ] + [pv.bbox for pv in visibility.props if pv.visible and pv.bbox]
            loc_dest = crop_dir / f"loc__{shot.location_id}.png"
            anchor = extract_location_anchor(
                chosen, subject_bboxes, loc_dest,
                model_name=settings.segment_model,
                bg_color=settings.segment_bg_color,
            )
        else:
            anchor = chosen
        memory.set_location(shot.location_id, anchor)

    for pid, state in shot.prop_states.items():
        if state in {PropState.not_visible, PropState.not_present}:
            continue
        pv = prop_visibility.get(pid)
        if visibility.props and (pv is None or not pv.visible):
            continue
        if pv and pv.bbox:
            anchor = _extract_anchor(
                chosen, pv.bbox, crop_dir / f"prop__{pid}__{state}.png", settings
            )
        else:
            anchor = chosen
        memory.set_prop(pid, state, anchor)
