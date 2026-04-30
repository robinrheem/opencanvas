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
    bg_prop_state_snapshot,
    crop_to_anchor,
    extract_visibility,
    generate,
    plan,
    render_location_anchor,
    render_scene_description,
    retrieve,
    segment_to_anchor,
    select,
)
from .config import CACHE_TAG_GENERATE, Settings
from .logs import Logger, set_active
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
    logger: Logger | None = None,
) -> list[Path]:
    key = make_generation_key(shot, anchors, seed, settings.k_candidates)
    hit = cache.get(key)
    if hit and all(Path(p).exists() for p in hit):
        if logger is not None:
            logger.log_cache(shot.index, "HIT", key)
        return [Path(p) for p in hit]
    if logger is not None:
        logger.log_cache(shot.index, "MISS", key)
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
            device=settings.device,
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


async def _seed_location_canonicals(
    p: Plan, memory: Memory, settings: Settings, pipe: ImagePipeline,
    bg_by_shot: dict[int, BackgroundPlan],
) -> None:
    """For each unique location_id, render one canonical empty-scene image
    from text only (no refs). Uses the FIRST shot in that location to seed
    bg-prop states. Snapshot stored in memory.location_canonical_state so
    later shots can detect deltas and re-render."""
    char_names = [c.name for c in p.characters]
    loc_name_by_id = {lo.id: lo.name for lo in p.locations}
    seen: set[str] = set()
    for shot in p.shots:
        lid = shot.location_id
        if not lid or lid in seen:
            continue
        seen.add(lid)
        bg_state = bg_prop_state_snapshot(shot, bg_by_shot.get(shot.index))
        scene = await render_scene_description(
            location_id=lid, location_name=loc_name_by_id.get(lid, lid),
            shots_in_loc=[s.description for s in p.shots if s.location_id == lid],
            character_names=char_names, bg_prop_states=bg_state, settings=settings,
        )
        # Deterministic seed per location for repeatability across runs.
        seed = settings.seed + hash(lid) % 10_000
        dest = memory.root / "locations" / f"{safe_filename(lid)}.png"
        render_location_anchor(scene, dest, settings, pipe, seed=seed)
        memory.locations[lid] = dest
        memory.location_canonical_state[lid] = bg_state


async def _maybe_refresh_location(
    shot: Shot, p: Plan, memory: Memory, settings: Settings, pipe: ImagePipeline,
    bg_by_shot: dict[int, BackgroundPlan], seed_of: Callable[[Shot], int],
) -> None:
    """Algorithm 4 location refresh: if bg-prop state changed since the last
    canonical render for this location_id, re-render the canonical with the
    new states embedded. Otherwise keep the existing anchor (no churn,
    consistent backgrounds across same-location shots)."""
    lid = shot.location_id
    if not lid:
        return
    bg_state = bg_prop_state_snapshot(shot, bg_by_shot.get(shot.index))
    if memory.location_canonical_state.get(lid) == bg_state:
        return
    char_names = [c.name for c in p.characters]
    loc_name_by_id = {lo.id: lo.name for lo in p.locations}
    scene = await render_scene_description(
        location_id=lid, location_name=loc_name_by_id.get(lid, lid),
        shots_in_loc=[s.description for s in p.shots if s.location_id == lid],
        character_names=char_names, bg_prop_states=bg_state, settings=settings,
    )
    dest = memory.root / "locations" / f"{safe_filename(lid)}.png"
    render_location_anchor(scene, dest, settings, pipe, seed=seed_of(shot))
    memory.locations[lid] = dest
    memory.location_canonical_state[lid] = bg_state


def _refresh_anchors(
    shot: Shot, chosen: Path, memory: Memory, visibility: FrameVisibility,
    crop_dir: Path, settings: Settings,
) -> None:
    """Algorithm 4 — VLM gates char/prop anchor refresh via bbox + segmentation.
    Location anchor refresh is handled separately by `_maybe_refresh_location`
    (text-to-image regen, not inpaint-from-frame)."""
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
    logger = Logger(settings.out_dir, enabled=settings.enable_logging)
    set_active(logger)

    p = await plan(story, settings)
    (settings.out_dir / "plan.json").write_text(p.model_dump_json(indent=2))
    bg_by_shot = {bp.shot_index: bp for bp in p.background_plans}

    # Pre-loop canonical location render: one empty-scene image per unique
    # location_id, generated text-to-image with current bg-prop states baked
    # into the prompt. Replaces inpaint-from-chosen-frame for cleaner anchors.
    await _seed_location_canonicals(p, memory, settings, pipe, bg_by_shot)

    results: list[ShotResult] = []
    with Cache(str(settings.cache_dir), tag_index=True) as cache:
        for shot in p.shots:
            anchors = retrieve(shot, p, memory)
            seed = seed_of(shot)
            chars_before = set(memory.characters.keys())
            props_before = set(memory.props.keys())
            locs_before = set(memory.locations.keys())

            candidates = _cached_generate(
                cache, shot, anchors, bg_by_shot.get(shot.index),
                settings, pipe, seed=seed, logger=logger,
            )
            chosen, scores = await select(candidates, shot, memory, settings)

            memory.add_frame(shot.index, chosen)
            visibility = await extract_visibility(shot, chosen, settings)
            _refresh_anchors(
                shot, chosen, memory, visibility,
                crop_dir=settings.out_dir / "crops" / f"shot_{shot.index:04d}",
                settings=settings,
            )
            await _maybe_refresh_location(
                shot, p, memory, settings, pipe, bg_by_shot, seed_of,
            )
            memory.save()

            best = max(scores, key=lambda s: s.overall_score)
            anchor_updates = sorted(
                [f"char {a}::{b}" for (a, b) in memory.characters.keys() - chars_before]
                + [f"prop {a}::{b}" for (a, b) in memory.props.keys() - props_before]
                + [f"location {lid}" for lid in memory.locations.keys() - locs_before]
            )
            logger.log_decision(shot.index, {
                "shot_index": shot.index,
                "description": shot.description,
                "location_id": shot.location_id,
                "continuation_mode": shot.continuation_mode.value,
                "expected_cast": [
                    cid for cid, st in shot.character_states.items() if st != "not_present"
                ],
                "anchors_fed": anchors.model_dump(),
                "candidate_paths": [str(c) for c in candidates],
                "candidate_seeds": [seed + i for i in range(settings.k_candidates)],
                "judge_scores": [s.model_dump() for s in scores],
                "chosen_index": best.candidate_index,
                "chosen": str(chosen),
                "visibility_after": visibility.model_dump(),
                "anchor_updates": anchor_updates,
            })

            results.append(ShotResult(
                shot=shot, anchors=anchors, candidates=candidates,
                scores=scores, chosen=chosen,
            ))
            _write_results(settings.out_dir, results)

    set_active(None)
    return p, results


def run_sync(
    story: Story, settings: Settings,
    image_pipeline: ImagePipeline | None = None,
    seed_maker: Callable[[Shot], int] | None = None,
) -> tuple[Plan, list[ShotResult]]:
    """Synchronous wrapper for `run`."""
    return asyncio.run(run(story, settings, image_pipeline, seed_maker))
