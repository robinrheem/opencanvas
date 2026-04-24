"""Algorithm 1 — CANVAS sequential generation with memory + diskcache resume."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from diskcache import Cache
from pydantic import TypeAdapter

from .agents import ImagePipeline, generate, plan_async, retrieve, select_async
from .config import CACHE_TAG_GENERATE, Settings
from .memory import Memory
from .schemas import (
    AnchorSet,
    CandidateScore,
    CharacterState,
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
    settings: Settings,
    pipeline: ImagePipeline,
    seed: int,
) -> list[Path]:
    key = make_generation_key(shot, anchors, seed, settings.k_candidates)
    hit = cache.get(key)
    if hit and all(Path(p).exists() for p in hit):
        return [Path(p) for p in hit]
    paths = generate(shot, anchors, settings, pipeline, seed=seed)
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

    results: list[ShotResult] = []
    with Cache(str(settings.cache_dir), tag_index=True) as cache:
        for shot in p.shots:
            anchors = retrieve(shot, memory)
            candidates = _cached_generate(cache, shot, anchors, settings, pipe, seed=seed_of(shot))
            chosen, scores = await select_async(candidates, shot, memory, settings)

            memory.set_frame(shot.index, chosen)
            _update_anchors_from_frame(shot, chosen, memory)
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


def _update_anchors_from_frame(shot: Shot, chosen: Path, memory: Memory) -> None:
    """Algorithm 4 (simplified): reuse the selected frame as the anchor.

    Paper's Algorithm 4 calls a VLM with Tables 27-29 to extract clean crops.
    """
    for cid, state in shot.character_states.items():
        if state == CharacterState.not_present:
            continue
        memory.set_character(cid, state, chosen)

    if shot.location_id:
        memory.set_location(shot.location_id, chosen)

    for pid, state in shot.prop_states.items():
        if state in {PropState.not_visible, PropState.not_present}:
            continue
        memory.set_prop(pid, state, chosen)
