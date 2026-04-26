"""The CANVAS agents — paper-faithful split.

Planner is four sub-calls (Tables 20, 22, 23, 24). Judge uses Table 26 axes.

Internals are async and dispatched concurrently via `asyncio.gather`. Public
sync wrappers (`plan`, `select`) drive an event loop via `asyncio.run` for
callers that prefer a synchronous API.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Protocol, TypeVar

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
    CandidateScore,
    Character,
    CharacterState,
    CharacterTimeline,
    ContinuationDecision,
    ContinuationMode,
    FrameVisibility,
    Location,
    LocationClustering,
    Plan,
    Prop,
    PropState,
    PropTimeline,
    Shot,
    Story,
)

T = TypeVar("T", bound=BaseModel)
_GRAY_FALLBACK_SIZE = 1024


class ImagePipeline(Protocol):
    def __call__(
        self,
        *,
        prompt: str,
        image: list,
        num_inference_steps: int,
        true_cfg_scale: float,
        guidance_scale: float,
        negative_prompt: str,
        generator,
    ): ...


@lru_cache(maxsize=8)
def _model_cached(model: str, base_url: str, api_key: str) -> OpenAIChatModel:
    return OpenAIChatModel(
        model, provider=OpenAIProvider(base_url=base_url, api_key=api_key)
    )


def _model(settings: Settings) -> OpenAIChatModel:
    return _model_cached(settings.model, settings.base_url, settings.api_key)


@lru_cache(maxsize=16)
def _agent_cached(
    model: str, base_url: str, api_key: str, output_type: type, instructions: str
) -> Agent:
    return Agent(
        _model_cached(model, base_url, api_key),
        output_type=output_type,
        instructions=instructions,
    )


def _agent_for(output_type: type[T], instructions: str, settings: Settings) -> Agent:
    return _agent_cached(
        settings.model, settings.base_url, settings.api_key, output_type, instructions
    )


def _numbered(items: list[str], prefix: str = "") -> str:
    return "\n".join(f"{prefix}{i + 1}. {s}" for i, s in enumerate(items))


def _pad(values: list, n: int, default) -> list:
    return values if len(values) == n else [default] * n


async def _run_planner_async(
    output_type: type[T], instructions: str, user_prompt: str, settings: Settings
) -> T:
    agent = _agent_for(output_type, instructions, settings)
    return (await agent.run(user_prompt)).output


# --- Planner sub-agents (async) ---------------------------------------------


async def _cluster_locations(story: Story, settings: Settings) -> LocationClustering:
    user = (
        f"Title: {story.title}\n\nShots:\n"
        + _numbered(story.shots, prefix="Scene_1_Shot_").replace(". ", ": ", 1)
        + f"\n\nKnown locations (hints): {[l.name for l in story.locations] or 'none'}\n"
        f"Return shot_location as a list aligned with the {len(story.shots)} shots above."
    )
    c = await _run_planner_async(LocationClustering, LOCATION_CLUSTERING, user, settings)
    c.shot_location = _pad(c.shot_location, len(story.shots), "loc-0")
    return c


async def _plan_character(
    char: Character, shots: list[str], settings: Settings
) -> CharacterTimeline:
    user = (
        f"Character: {char.name}\nDescription: {char.description}\n\n"
        f"Shots:\n{_numbered(shots)}\n\n"
        f"Return appearance_by_shot as a list of length {len(shots)}."
    )
    tl = await _run_planner_async(CharacterTimeline, CHARACTER_PLANNING, user, settings)
    tl.character_id = char.id
    tl.appearance_by_shot = _pad(tl.appearance_by_shot, len(shots), CharacterState.default)
    return tl


async def _plan_prop(prop: Prop, shots: list[str], settings: Settings) -> PropTimeline:
    user = (
        f"Prop: {prop.name}\nDescription: {prop.description}\n\n"
        f"Shots:\n{_numbered(shots)}\n\n"
        f"Return state_by_shot and carrier_by_shot, each of length {len(shots)}."
    )
    tl = await _run_planner_async(PropTimeline, PROP_PLANNING, user, settings)
    tl.prop_id = prop.id
    tl.state_by_shot = _pad(tl.state_by_shot, len(shots), PropState.not_visible)
    tl.carrier_by_shot = _pad(tl.carrier_by_shot, len(shots), None)
    return tl


async def _decide_continuation(
    prev_desc: str, curr_desc: str, prev_loc: str, curr_loc: str, settings: Settings
) -> ContinuationMode:
    user = (
        f"Previous shot: {prev_desc}\nPrevious location: {prev_loc}\n\n"
        f"Current shot: {curr_desc}\nCurrent location: {curr_loc}\n\n"
        "Decide continuation_mode."
    )
    decision = await _run_planner_async(
        ContinuationDecision, CONTINUATION_DECISION, user, settings
    )
    return decision.continuation_mode


async def _plan_background(
    shot_index: int,
    shot_description: str,
    shot_metadata: dict,
    prop_history: list[dict],
    settings: Settings,
) -> BackgroundPlan:
    user = (
        BACKGROUND_PLANNING.format(
            shot_description=shot_description,
            shot_metadata=shot_metadata,
            prop_history=prop_history or "(none)",
        )
        + f"\n\nReturn a BackgroundPlan for shot index {shot_index}."
    )
    bp = await _run_planner_async(BackgroundPlan, BACKGROUND_PLANNING, user, settings)
    bp.shot_index = shot_index
    return bp


# --- Main planner ------------------------------------------------------------


async def plan_async(story: Story, settings: Settings) -> Plan:
    """Global Planner Agent (§3.1) = Tables 20 + 22 + 23 + 24 + 21, gathered."""
    cluster = await _cluster_locations(story, settings)

    char_task = asyncio.gather(
        *(_plan_character(c, story.shots, settings) for c in story.characters)
    )
    prop_task = asyncio.gather(*(_plan_prop(p, story.shots, settings) for p in story.props))
    continuation_task = asyncio.gather(
        *(
            _decide_continuation(
                prev_desc=story.shots[t - 1],
                curr_desc=story.shots[t],
                prev_loc=cluster.shot_location[t - 1],
                curr_loc=cluster.shot_location[t],
                settings=settings,
            )
            for t in range(1, len(story.shots))
        )
    )
    char_results, prop_results, raw_continuations = await asyncio.gather(
        char_task, prop_task, continuation_task
    )

    char_timelines = {tl.character_id: tl.appearance_by_shot for tl in char_results}
    prop_timelines = {tl.prop_id: tl.state_by_shot for tl in prop_results}
    prop_carriers = {tl.prop_id: tl.carrier_by_shot for tl in prop_results}

    continuations: list[ContinuationMode] = [ContinuationMode.fresh_location]
    seen_locations: set[str] = {cluster.shot_location[0]}
    for t, mode in enumerate(raw_continuations, start=1):
        if mode is ContinuationMode.location_reappearance and cluster.shot_location[t] not in seen_locations:
            mode = ContinuationMode.fresh_location
        continuations.append(mode)
        seen_locations.add(cluster.shot_location[t])

    shots = [
        Shot(
            index=i,
            description=desc,
            location_id=cluster.shot_location[i],
            continuation_mode=continuations[i],
            character_states={cid: tl[i] for cid, tl in char_timelines.items()},
            prop_states={pid: tl[i] for pid, tl in prop_timelines.items()},
            prop_carriers={
                pid: carriers[i]
                for pid, carriers in prop_carriers.items()
                if carriers[i]
            },
        )
        for i, desc in enumerate(story.shots)
    ]

    background_plans = await asyncio.gather(
        *(
            _plan_background(
                shot_index=s.index,
                shot_description=s.description,
                shot_metadata={
                    "characters": s.character_states,
                    "location": s.location_id,
                    "props": s.prop_states,
                },
                prop_history=[
                    {"shot": prev.index, "props": prev.prop_states}
                    for prev in shots[: s.index]
                    if prev.prop_states
                ],
                settings=settings,
            )
            for s in shots
        )
    )

    known = {l.id for l in story.locations}
    extra_locs = [
        Location(id=lid, name=cluster.location_names.get(lid, lid), description="")
        for lid in set(cluster.shot_location) - known
    ]
    return Plan(
        shots=shots,
        characters=story.characters,
        locations=[*story.locations, *extra_locs],
        props=story.props,
        background_plans=list(background_plans),
    )


def plan(story: Story, settings: Settings) -> Plan:
    """Synchronous wrapper for `plan_async`."""
    return asyncio.run(plan_async(story, settings))


# --- Anchor retrieval (Algorithm 2) -----------------------------------------


def _collect_refs(states: dict[str, str], getter, skip: set[str]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for entity_id, state in states.items():
        if state in skip:
            continue
        anchor = getter(entity_id, state)
        if anchor is None:
            continue
        s = str(anchor)
        if s in seen:
            continue
        refs.append(s)
        seen.add(s)
    return refs


def _character_anchor(
    memory: Memory, char_id: str, current_state: str, prev_state: str | None
) -> Path | None:
    """Algorithm 2 §4: appearance change → canonical first; else → recent first."""
    canonical = memory.get_character_canonical(char_id, current_state)
    recent = memory.characters.get((char_id, current_state))
    if prev_state is None or prev_state != current_state:
        return canonical or recent
    return recent or canonical


def retrieve(shot: Shot, plan: Plan, memory: Memory) -> AnchorSet:
    prev_shot = plan.shots[shot.index - 1] if shot.index > 0 else None

    def char_getter(cid: str, state: str) -> Path | None:
        prev_state = prev_shot.character_states.get(cid) if prev_shot else None
        return _character_anchor(memory, cid, state, prev_state)

    char_refs = _collect_refs(
        shot.character_states, char_getter, {CharacterState.not_present}
    )
    prop_refs = _collect_refs(
        shot.prop_states,
        lambda pid, state: (
            memory.get_prop(pid, state)
            or memory.get_prop(pid, PropState.default)
            or memory.get_prop_any_state(pid)  # paper §3.2 Alg.2 step 6 — any prior state
        ),
        {PropState.not_visible, PropState.not_present},
    )
    anchors = AnchorSet(character_refs=char_refs, prop_refs=prop_refs)

    if shot.continuation_mode is ContinuationMode.previous_frame_continuation:
        prev = memory.get_frame(shot.index - 1)
        anchors.previous_frame = str(prev) if prev else None
    elif shot.continuation_mode is ContinuationMode.location_reappearance and shot.location_id:
        loc = memory.get_location(shot.location_id)
        anchors.location_ref = str(loc) if loc else None

    return anchors


# --- Image generation (Algorithm 2 step 7 + Table 25) -----------------------


def _format_states(states: dict[str, str]) -> str:
    if not states:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in states.items())


def _format_prop_states_with_carriers(
    states: dict[str, str], carriers: dict[str, str]
) -> str:
    if not states:
        return "(none)"
    parts = []
    for pid, state in states.items():
        carrier = carriers.get(pid)
        parts.append(f"{pid}={state}" + (f" (carried by {carrier})" if carrier else ""))
    return ", ".join(parts)


def crop_to_anchor(frame_path: Path, bbox, dest: Path) -> Path:
    """Crop a normalized BBox out of a frame and save to dest."""
    from PIL import Image

    with Image.open(frame_path) as im:
        w, h = im.size
        left = max(0, int(bbox.x * w))
        top = max(0, int(bbox.y * h))
        right = min(w, int((bbox.x + bbox.w) * w))
        bottom = min(h, int((bbox.y + bbox.h) * h))
        if right <= left or bottom <= top:
            # Degenerate bbox; fall back to full frame.
            im.convert("RGB").save(dest)
        else:
            im.crop((left, top, right, bottom)).convert("RGB").save(dest)
    return dest


@lru_cache(maxsize=2)
def _rembg_session(model_name: str):
    from rembg import new_session

    return new_session(model_name)


def segment_to_anchor(
    frame_path: Path,
    bbox,
    dest: Path,
    model_name: str = "birefnet-general",
    bg_color: tuple[int, int, int] = (128, 128, 128),
) -> Path:
    """Bbox-crop a region, then segment subject and composite onto neutral bg.

    Mitigates the background-drift problem when reference images are passed to
    multi-image edit models (Qwen-Image-Edit-2509 etc.): without segmentation,
    the reference's background pixels condition the generator alongside the
    subject, causing the output to inherit the reference's setting.
    """
    from PIL import Image
    from rembg import remove

    with Image.open(frame_path) as im:
        w, h = im.size
        left = max(0, int(bbox.x * w))
        top = max(0, int(bbox.y * h))
        right = min(w, int((bbox.x + bbox.w) * w))
        bottom = min(h, int((bbox.y + bbox.h) * h))
        if right <= left or bottom <= top:
            im.convert("RGB").save(dest)
            return dest
        crop = im.crop((left, top, right, bottom)).convert("RGB")

    session = _rembg_session(model_name)
    foreground = remove(crop, session=session, post_process_mask=True)
    if foreground.mode != "RGBA":
        foreground = foreground.convert("RGBA")

    bg = Image.new("RGBA", foreground.size, (*bg_color, 255))
    composite = Image.alpha_composite(bg, foreground).convert("RGB")
    composite.save(dest)
    return dest


def _load_refs(paths: list[str]):
    from PIL import Image

    imgs = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    if not imgs:
        imgs.append(Image.new("RGB", (_GRAY_FALLBACK_SIZE, _GRAY_FALLBACK_SIZE), (128, 128, 128)))
    return imgs


def _anchor_labels(anchors: AnchorSet) -> list[str]:
    return (
        (["previous frame (spatial continuity)"] if anchors.previous_frame else [])
        + ["character anchor"] * len(anchors.character_refs)
        + (["background anchor"] if anchors.location_ref else [])
        + ["prop anchor"] * len(anchors.prop_refs)
    )


def _format_background_plan(bp: BackgroundPlan | None) -> str:
    if bp is None:
        return "(no background plan)"
    parts = []
    if bp.must_appear:
        parts.append(f"must_appear={bp.must_appear}")
    if bp.must_not_appear:
        parts.append(f"must_not_appear={bp.must_not_appear}")
    if bp.background_props:
        parts.append(f"persistent_bg_props={bp.background_props}")
    if bp.carried_props:
        parts.append(f"carried_props={bp.carried_props}")
    return "; ".join(parts) or "(no constraints)"


def generate(
    shot: Shot,
    anchors: AnchorSet,
    background_plan: BackgroundPlan | None,
    settings: Settings,
    pipeline: ImagePipeline,
    seed: int,
) -> list[Path]:
    # lazy: tests stub sys.modules['torch'] before this runs
    import torch

    ref_imgs = _load_refs(anchors.ordered_paths())
    prompt = CANDIDATE_GENERATION.format(
        shot_description=shot.description,
        anchor_summary=", ".join(_anchor_labels(anchors)) or "(none)",
        character_states=_format_states(shot.character_states),
        prop_states=_format_prop_states_with_carriers(shot.prop_states, shot.prop_carriers),
        location=shot.location_id or "(unknown)",
    ) + f"\n\nBackground constraints: {_format_background_plan(background_plan)}"
    out_dir = settings.out_dir / "candidates" / f"shot_{shot.index:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for i in range(settings.k_candidates):
        result = pipeline(
            prompt=prompt,
            image=ref_imgs,
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


async def _score_candidate(
    agent: Agent, parts: list, index: int
) -> CandidateScore:
    s = (await agent.run(parts)).output
    s.candidate_index = index
    return s


async def select_async(
    candidates: list[Path],
    shot: Shot,
    memory: Memory,
    settings: Settings,
) -> tuple[Path, list[CandidateScore]]:
    agent = _agent_for(CandidateScore, JUDGE_SCORING, settings)
    judge_prompt = JUDGE_SCORING.format(
        shot_description=shot.description,
        character_states=_format_states(shot.character_states),
        prop_states=_format_states(shot.prop_states),
        location=shot.location_id or "(unknown)",
    )
    prev_frame = memory.get_frame(shot.index - 1)
    prev_part = (
        ["Previous frame:", BinaryContent.from_path(prev_frame)] if prev_frame else []
    )

    tasks = [
        _score_candidate(
            agent,
            [
                judge_prompt,
                f"Candidate index: {i}",
                BinaryContent.from_path(cand),
                *prev_part,
            ],
            i,
        )
        for i, cand in enumerate(candidates)
    ]
    scores = await asyncio.gather(*tasks)
    best = max(scores, key=lambda s: s.overall_score)
    return candidates[best.candidate_index], list(scores)


def select(
    candidates: list[Path],
    shot: Shot,
    memory: Memory,
    settings: Settings,
) -> tuple[Path, list[CandidateScore]]:
    """Synchronous wrapper for `select_async`."""
    return asyncio.run(select_async(candidates, shot, memory, settings))


# --- Algorithm 4 — VLM visibility gating (Tables 27, 28, 29) ----------------

from .schemas import CharacterVisibility, PropVisibility


class _CharVisibilityResp(BaseModel):
    characters: list[CharacterVisibility]


class _PropVisibilityResp(BaseModel):
    props: list[PropVisibility]


class _LocVisibilityResp(BaseModel):
    visible: bool


async def _vlm_check(
    response_type: type[T], instructions: str, frame: Path, settings: Settings
) -> T:
    agent = _agent_for(response_type, instructions, settings)
    res = await agent.run(["Inspect the frame.", BinaryContent.from_path(frame)])
    return res.output


async def _check_characters_visible(
    shot: Shot, frame: Path, settings: Settings
) -> list[CharacterVisibility]:
    expected = {
        cid: state
        for cid, state in shot.character_states.items()
        if state != CharacterState.not_present
    }
    if not expected:
        return []
    instructions = CHAR_VISIBILITY.format(
        shot_description=shot.description, expected_characters=expected
    )
    return (await _vlm_check(_CharVisibilityResp, instructions, frame, settings)).characters


async def _check_props_visible(
    shot: Shot, frame: Path, settings: Settings
) -> list[PropVisibility]:
    expected = {
        pid: state
        for pid, state in shot.prop_states.items()
        if state not in {PropState.not_visible, PropState.not_present}
    }
    if not expected:
        return []
    instructions = PROP_VISIBILITY.format(
        shot_description=shot.description, expected_props=expected
    )
    return (await _vlm_check(_PropVisibilityResp, instructions, frame, settings)).props


async def _check_location_visible(shot: Shot, frame: Path, settings: Settings) -> bool:
    if not shot.location_id:
        return False
    instructions = BG_VISIBILITY.format(
        shot_description=shot.description, location=shot.location_id
    )
    return (await _vlm_check(_LocVisibilityResp, instructions, frame, settings)).visible


async def extract_visibility_async(
    shot: Shot, chosen: Path, settings: Settings
) -> FrameVisibility:
    chars, location_ok, props = await asyncio.gather(
        _check_characters_visible(shot, chosen, settings),
        _check_location_visible(shot, chosen, settings),
        _check_props_visible(shot, chosen, settings),
    )
    return FrameVisibility(
        characters=chars, location_visible=location_ok, props=props
    )


def extract_visibility(
    shot: Shot, chosen: Path, settings: Settings
) -> FrameVisibility:
    return asyncio.run(extract_visibility_async(shot, chosen, settings))
