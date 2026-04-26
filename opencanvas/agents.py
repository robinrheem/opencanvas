"""The CANVAS agents — paper-faithful split.

Planner is four sub-calls (Tables 20, 22, 23, 24) plus background plan (Table 21).
Judge uses Table 26 axes. Visibility extraction uses Tables 27, 28, 29.

All LLM-touching functions are async by default (`plan`, `select`,
`extract_visibility`). `plan_sync` exists as a thin sync wrapper for the CLI;
everything else awaits via `asyncio.gather` from `pipeline.run`.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Protocol, TypeVar

from PIL import Image
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Settings
from .memory import Memory
from .prompts import (
    BACKGROUND_PLANNING,
    BG_VISIBILITY,
    CANDIDATE_GENERATION,
    CHAR_VISIBILITY,
    CHARACTER_PLANNING,
    CONTINUATION_DECISION,
    JUDGE_SCORING,
    LOCATION_CLUSTERING,
    PROP_PLANNING,
    PROP_VISIBILITY,
)
from .schemas import (
    AnchorSet,
    BackgroundPlan,
    BBox,
    CandidateScore,
    Character,
    CharacterState,
    CharacterTimeline,
    CharacterVisibility,
    ContinuationDecision,
    ContinuationMode,
    FrameVisibility,
    Location,
    LocationClustering,
    Plan,
    Prop,
    PropState,
    PropTimeline,
    PropVisibility,
    Shot,
    Story,
)

T = TypeVar("T", bound=BaseModel)
_GRAY = (128, 128, 128)
_FALLBACK_SIZE = 1024


class ImagePipeline(Protocol):
    def __call__(
        self, *, prompt: str, image: list, num_inference_steps: int,
        true_cfg_scale: float, guidance_scale: float, negative_prompt: str, generator,
    ): ...


# --- Model + agent caching -------------------------------------------------


@lru_cache(maxsize=8)
def _model_cached(model: str, base_url: str, api_key: str) -> OpenAIChatModel:
    return OpenAIChatModel(model, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


@lru_cache(maxsize=16)
def _agent_cached(
    model: str, base_url: str, api_key: str, output_type: type, instructions: str
) -> Agent:
    return Agent(_model_cached(model, base_url, api_key), output_type=output_type, instructions=instructions)


def _agent_for(output_type: type[T], instructions: str, settings: Settings) -> Agent:
    return _agent_cached(settings.model, settings.base_url, settings.api_key, output_type, instructions)


async def _llm(output_type: type[T], instructions: str, prompt, settings: Settings) -> T:
    """Run a typed pydantic-ai call. `prompt` is a string or list of parts."""
    return (await _agent_for(output_type, instructions, settings).run(prompt)).output


# --- Formatting helpers ------------------------------------------------------


def _numbered(items: list[str], prefix: str = "") -> str:
    return "\n".join(f"{prefix}{i + 1}. {s}" for i, s in enumerate(items))


def _pad(values: list, n: int, default) -> list:
    return values if len(values) == n else [default] * n


def _format_states(states: dict[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in states.items()) or "(none)"


def _format_prop_states_with_carriers(states: dict[str, str], carriers: dict[str, str]) -> str:
    if not states:
        return "(none)"
    return ", ".join(
        f"{pid}={state}" + (f" (carried by {carriers[pid]})" if pid in carriers else "")
        for pid, state in states.items()
    )


def _format_background_plan(bp: BackgroundPlan | None) -> str:
    if bp is None:
        return "(no background plan)"
    parts = [
        f"{label}={value}"
        for label, value in (
            ("must_appear", bp.must_appear),
            ("must_not_appear", bp.must_not_appear),
            ("persistent_bg_props", bp.background_props),
            ("carried_props", bp.carried_props),
        )
        if value
    ]
    return "; ".join(parts) or "(no constraints)"


# --- Planner sub-agents (async) ---------------------------------------------


async def _cluster_locations(story: Story, settings: Settings) -> LocationClustering:
    user = (
        f"Title: {story.title}\n\nShots:\n"
        + _numbered(story.shots, prefix="Scene_1_Shot_").replace(". ", ": ", 1)
        + f"\n\nKnown locations (hints): {[l.name for l in story.locations] or 'none'}\n"
        f"Return shot_location as a list aligned with the {len(story.shots)} shots above."
    )
    c = await _llm(LocationClustering, LOCATION_CLUSTERING, user, settings)
    c.shot_location = _pad(c.shot_location, len(story.shots), "loc-0")
    return c


async def _plan_character(char: Character, shots: list[str], settings: Settings) -> list[str]:
    user = (
        f"Character: {char.name}\nDescription: {char.description}\n\n"
        f"Shots:\n{_numbered(shots)}\n\n"
        f"Return appearance_by_shot as a list of length {len(shots)}."
    )
    tl = await _llm(CharacterTimeline, CHARACTER_PLANNING, user, settings)
    return _pad(tl.appearance_by_shot, len(shots), CharacterState.default)


async def _plan_prop(prop: Prop, shots: list[str], settings: Settings) -> tuple[list[str], list[str | None]]:
    user = (
        f"Prop: {prop.name}\nDescription: {prop.description}\n\n"
        f"Shots:\n{_numbered(shots)}\n\n"
        f"Return state_by_shot and carrier_by_shot, each of length {len(shots)}."
    )
    tl = await _llm(PropTimeline, PROP_PLANNING, user, settings)
    return (
        _pad(tl.state_by_shot, len(shots), PropState.not_visible),
        _pad(tl.carrier_by_shot, len(shots), None),
    )


async def _decide_continuation(
    prev_desc: str, curr_desc: str, prev_loc: str, curr_loc: str, settings: Settings
) -> ContinuationMode:
    user = (
        f"Previous shot: {prev_desc}\nPrevious location: {prev_loc}\n\n"
        f"Current shot: {curr_desc}\nCurrent location: {curr_loc}\n\n"
        "Decide continuation_mode."
    )
    return (await _llm(ContinuationDecision, CONTINUATION_DECISION, user, settings)).continuation_mode


async def _plan_background(shot: Shot, prop_history: list[dict], settings: Settings) -> BackgroundPlan:
    user = (
        BACKGROUND_PLANNING.format(
            shot_description=shot.description,
            shot_metadata={
                "characters": shot.character_states,
                "location": shot.location_id,
                "props": shot.prop_states,
            },
            prop_history=prop_history or "(none)",
        )
        + f"\n\nReturn a BackgroundPlan for shot index {shot.index}."
    )
    bp = await _llm(BackgroundPlan, BACKGROUND_PLANNING, user, settings)
    bp.shot_index = shot.index
    return bp


# --- Main planner ------------------------------------------------------------


async def plan(story: Story, settings: Settings) -> Plan:
    """Global Planner Agent (§3.1) = Tables 20 + 22 + 23 + 24 + 21, gathered."""
    cluster = await _cluster_locations(story, settings)

    char_results, prop_results, raw_continuations = await asyncio.gather(
        asyncio.gather(*(_plan_character(c, story.shots, settings) for c in story.characters)),
        asyncio.gather(*(_plan_prop(p, story.shots, settings) for p in story.props)),
        asyncio.gather(*(
            _decide_continuation(
                prev_desc=story.shots[t - 1], curr_desc=story.shots[t],
                prev_loc=cluster.shot_location[t - 1], curr_loc=cluster.shot_location[t],
                settings=settings,
            )
            for t in range(1, len(story.shots))
        )),
    )

    char_timelines = dict(zip([c.id for c in story.characters], char_results))
    prop_timelines = dict(zip([p.id for p in story.props], (s for s, _ in prop_results)))
    prop_carriers = dict(zip([p.id for p in story.props], (c for _, c in prop_results)))

    continuations: list[ContinuationMode] = [ContinuationMode.fresh_location]
    seen: set[str] = {cluster.shot_location[0]}
    for t, mode in enumerate(raw_continuations, start=1):
        if mode is ContinuationMode.location_reappearance and cluster.shot_location[t] not in seen:
            mode = ContinuationMode.fresh_location
        continuations.append(mode)
        seen.add(cluster.shot_location[t])

    shots = [
        Shot(
            index=i, description=desc, location_id=cluster.shot_location[i],
            continuation_mode=continuations[i],
            character_states={cid: tl[i] for cid, tl in char_timelines.items()},
            prop_states={pid: tl[i] for pid, tl in prop_timelines.items()},
            prop_carriers={pid: c[i] for pid, c in prop_carriers.items() if c[i]},
        )
        for i, desc in enumerate(story.shots)
    ]

    background_plans = await asyncio.gather(*(
        _plan_background(
            s,
            [{"shot": prev.index, "props": prev.prop_states} for prev in shots[: s.index] if prev.prop_states],
            settings,
        )
        for s in shots
    ))

    known = {l.id for l in story.locations}
    extra_locs = [
        Location(id=lid, name=cluster.location_names.get(lid, lid), description="")
        for lid in set(cluster.shot_location) - known
    ]
    return Plan(
        shots=shots, characters=story.characters,
        locations=[*story.locations, *extra_locs],
        props=story.props, background_plans=list(background_plans),
    )


def plan_sync(story: Story, settings: Settings) -> Plan:
    """Synchronous wrapper for `plan` (CLI convenience)."""
    return asyncio.run(plan(story, settings))


# --- Anchor retrieval (Algorithm 2) -----------------------------------------


def _collect_refs(states: dict[str, str], getter, skip: set[str]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for entity_id, state in states.items():
        if state in skip:
            continue
        anchor = getter(entity_id, state)
        if anchor is None or str(anchor) in seen:
            continue
        refs.append(str(anchor))
        seen.add(str(anchor))
    return refs


def retrieve(shot: Shot, plan: Plan, memory: Memory) -> AnchorSet:
    prev_shot = plan.shots[shot.index - 1] if shot.index > 0 else None

    def char_anchor(cid: str, state: str) -> Path | None:
        """Algorithm 2 §4: appearance changed → canonical first; else recent first."""
        prev_state = prev_shot.character_states.get(cid) if prev_shot else None
        canonical = memory.characters_canonical.get((cid, state))
        recent = memory.characters.get((cid, state))
        if prev_state is None or prev_state != state:
            return canonical or recent
        return recent or canonical

    char_refs = _collect_refs(shot.character_states, char_anchor, {CharacterState.not_present})
    prop_refs = _collect_refs(
        shot.prop_states, memory.prop, {PropState.not_visible, PropState.not_present}
    )
    anchors = AnchorSet(character_refs=char_refs, prop_refs=prop_refs)

    if shot.continuation_mode is ContinuationMode.previous_frame_continuation:
        prev = memory.frames.get(shot.index - 1)
        anchors.previous_frame = str(prev) if prev else None
    elif shot.continuation_mode is ContinuationMode.location_reappearance and shot.location_id:
        loc = memory.locations.get(shot.location_id)
        anchors.location_ref = str(loc) if loc else None

    return anchors


# --- Anchor extraction primitives ------------------------------------------


def _bbox_pixels(bbox: BBox, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    w, h = size
    left = max(0, int(bbox.x * w))
    top = max(0, int(bbox.y * h))
    right = min(w, int((bbox.x + bbox.w) * w))
    bottom = min(h, int((bbox.y + bbox.h) * h))
    return (left, top, right, bottom) if right > left and bottom > top else None


@lru_cache(maxsize=2)
def _rembg_session(model_name: str):
    from rembg import new_session
    return new_session(model_name)


def _segment_rgba(crop: Image.Image, model_name: str) -> Image.Image:
    """Returns RGBA image with subject pixels opaque, background transparent."""
    from rembg import remove
    fg = remove(crop.convert("RGB"), session=_rembg_session(model_name), post_process_mask=True)
    return fg if fg.mode == "RGBA" else fg.convert("RGBA")


def crop_to_anchor(frame_path: Path, bbox: BBox, dest: Path) -> Path:
    with Image.open(frame_path) as im:
        box = _bbox_pixels(bbox, im.size)
        (im.crop(box) if box else im).convert("RGB").save(dest)
    return dest


def segment_to_anchor(
    frame_path: Path, bbox: BBox, dest: Path,
    model_name: str = "birefnet-general", bg_color: tuple[int, int, int] = _GRAY,
) -> Path:
    """Bbox-crop, segment subject, composite onto neutral plate.

    Mitigates background-drift when reference images are passed to multi-image
    edit models: without segmentation, the reference's background pixels
    condition the generator alongside the subject.
    """
    with Image.open(frame_path) as im:
        box = _bbox_pixels(bbox, im.size)
        if not box:
            im.convert("RGB").save(dest)
            return dest
        crop = im.crop(box).convert("RGB")
    fg = _segment_rgba(crop, model_name)
    plate = Image.new("RGBA", fg.size, (*bg_color, 255))
    Image.alpha_composite(plate, fg).convert("RGB").save(dest)
    return dest


def extract_location_anchor(
    frame_path: Path, subject_bboxes: list[BBox], dest: Path,
    model_name: str = "birefnet-general", bg_color: tuple[int, int, int] = _GRAY,
) -> Path:
    """Background anchor: segment-out each subject silhouette, neutral-fill.

    Paper Table 28: "Avoid including large foreground characters when possible."
    Background geometry between subjects is preserved intact.
    """
    with Image.open(frame_path) as im:
        out = im.convert("RGBA").copy()
    if not subject_bboxes:
        out.convert("RGB").save(dest)
        return dest

    plate_full = Image.new("RGBA", out.size, (*bg_color, 255))
    for bbox in subject_bboxes:
        box = _bbox_pixels(bbox, out.size)
        if not box:
            continue
        region = out.crop(box)
        fg = _segment_rgba(region, model_name)
        plate = plate_full.crop(box)
        out.paste(Image.composite(plate, region, fg.split()[-1]), box[:2])

    out.convert("RGB").save(dest)
    return dest


# --- Image generation (Algorithm 2 step 7 + Table 25) -----------------------


def _load_refs(paths: list[str]) -> list[Image.Image]:
    imgs: list[Image.Image] = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    if not imgs:
        imgs.append(Image.new("RGB", (_FALLBACK_SIZE, _FALLBACK_SIZE), _GRAY))
    return imgs


def _anchor_labels(anchors: AnchorSet) -> str:
    labels = (
        (["previous frame (spatial continuity)"] if anchors.previous_frame else [])
        + ["character anchor"] * len(anchors.character_refs)
        + (["background anchor"] if anchors.location_ref else [])
        + ["prop anchor"] * len(anchors.prop_refs)
    )
    return ", ".join(labels) or "(none)"


def generate(
    shot: Shot, anchors: AnchorSet, background_plan: BackgroundPlan | None,
    settings: Settings, pipeline: ImagePipeline, seed: int,
) -> list[Path]:
    # lazy: tests stub sys.modules['torch'] before this runs
    import torch

    ref_imgs = _load_refs(anchors.ordered_paths())
    prompt = (
        CANDIDATE_GENERATION.format(
            shot_description=shot.description,
            anchor_summary=_anchor_labels(anchors),
            character_states=_format_states(shot.character_states),
            prop_states=_format_prop_states_with_carriers(shot.prop_states, shot.prop_carriers),
            location=shot.location_id or "(unknown)",
        )
        + f"\n\nBackground constraints: {_format_background_plan(background_plan)}"
    )
    out_dir = settings.out_dir / "candidates" / f"shot_{shot.index:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for i in range(settings.k_candidates):
        result = pipeline(
            prompt=prompt, image=ref_imgs,
            num_inference_steps=settings.num_inference_steps,
            true_cfg_scale=settings.true_cfg_scale,
            guidance_scale=settings.guidance_scale,
            negative_prompt=" ",
            generator=torch.Generator(device="cuda").manual_seed(seed + i),
        )
        p = out_dir / f"cand_{i}.png"
        result.images[0].save(p)
        paths.append(p)
    return paths


# --- QA-based selection (Algorithm 3 + Table 26) ----------------------------


async def select(
    candidates: list[Path], shot: Shot, memory: Memory, settings: Settings,
) -> tuple[Path, list[CandidateScore]]:
    judge_prompt = JUDGE_SCORING.format(
        shot_description=shot.description,
        character_states=_format_states(shot.character_states),
        prop_states=_format_states(shot.prop_states),
        location=shot.location_id or "(unknown)",
    )
    prev_frame = memory.frames.get(shot.index - 1)
    prev_part = ["Previous frame:", BinaryContent.from_path(prev_frame)] if prev_frame else []

    async def _score(i: int, cand: Path) -> CandidateScore:
        s = await _llm(
            CandidateScore, JUDGE_SCORING,
            [judge_prompt, f"Candidate index: {i}", BinaryContent.from_path(cand), *prev_part],
            settings,
        )
        s.candidate_index = i
        return s

    scores = list(await asyncio.gather(*(_score(i, c) for i, c in enumerate(candidates))))
    best = max(scores, key=lambda s: s.overall_score)
    return candidates[best.candidate_index], scores


# --- Algorithm 4 — VLM visibility extraction (Tables 27, 28, 29) -----------


class _CharVis(BaseModel):
    characters: list[CharacterVisibility]


class _PropVis(BaseModel):
    props: list[PropVisibility]


class _LocVis(BaseModel):
    visible: bool


async def extract_visibility(shot: Shot, chosen: Path, settings: Settings) -> FrameVisibility:
    """Three concurrent VLM checks (one per Tables 27/28/29) merged into a FrameVisibility."""
    parts = ["Inspect the frame.", BinaryContent.from_path(chosen)]

    async def chars() -> list[CharacterVisibility]:
        expected = {
            cid: st for cid, st in shot.character_states.items()
            if st != CharacterState.not_present
        }
        if not expected:
            return []
        instr = CHAR_VISIBILITY.format(shot_description=shot.description, expected_characters=expected)
        return (await _llm(_CharVis, instr, parts, settings)).characters

    async def props() -> list[PropVisibility]:
        expected = {
            pid: st for pid, st in shot.prop_states.items()
            if st not in {PropState.not_visible, PropState.not_present}
        }
        if not expected:
            return []
        instr = PROP_VISIBILITY.format(shot_description=shot.description, expected_props=expected)
        return (await _llm(_PropVis, instr, parts, settings)).props

    async def location() -> bool:
        if not shot.location_id:
            return False
        instr = BG_VISIBILITY.format(shot_description=shot.description, location=shot.location_id)
        return (await _llm(_LocVis, instr, parts, settings)).visible

    c, l, p = await asyncio.gather(chars(), location(), props())
    return FrameVisibility(characters=c, location_visible=l, props=p)
