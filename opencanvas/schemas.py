from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, Field


class ContinuationMode(StrEnum):
    previous_frame_continuation = "previous_frame_continuation"
    location_reappearance = "location_reappearance"
    fresh_location = "fresh_location"


class CharacterState(StrEnum):
    default = "default"
    not_present = "not_present"


class PropState(StrEnum):
    default = "default"
    not_visible = "not_visible"
    not_present = "not_present"


class Character(BaseModel):
    id: str
    name: str
    description: str
    appearance_states: list[str] = Field(default_factory=lambda: [CharacterState.default])
    reference_image: str | None = Field(
        default=None,
        description="Path to canonical reference image. Seeded into character memory for the 'default' state at run start.",
    )


class Location(BaseModel):
    id: str
    name: str
    description: str


class Prop(BaseModel):
    id: str
    name: str
    description: str
    states: list[str] = Field(default_factory=lambda: [PropState.default])


class Shot(BaseModel):
    index: int
    description: str
    location_id: str | None = None
    continuation_mode: ContinuationMode = ContinuationMode.fresh_location
    character_states: dict[str, str] = Field(default_factory=dict)
    prop_states: dict[str, str] = Field(default_factory=dict)


class LocationClustering(BaseModel):
    shot_location: list[str] = Field(
        description="location_id for each shot in order (length == num shots)."
    )
    location_names: dict[str, str] = Field(default_factory=dict)


class CharacterTimeline(BaseModel):
    character_id: str
    appearance_by_shot: list[str]


class PropTimeline(BaseModel):
    prop_id: str
    state_by_shot: list[str]
    carrier_by_shot: list[str | None] = Field(default_factory=list)


class ContinuationDecision(BaseModel):
    continuation_mode: ContinuationMode
    reasoning: str = ""


class Plan(BaseModel):
    shots: list[Shot]
    characters: list[Character]
    locations: list[Location]
    props: list[Prop] = Field(default_factory=list)


class AnchorSet(BaseModel):
    character_refs: list[str] = Field(default_factory=list)
    location_ref: str | None = None
    prop_refs: list[str] = Field(default_factory=list)
    previous_frame: str | None = None

    def ordered_paths(self) -> list[str]:
        out: list[str] = []
        if self.previous_frame:
            out.append(self.previous_frame)
        out.extend(self.character_refs)
        if self.location_ref:
            out.append(self.location_ref)
        out.extend(self.prop_refs)
        return out


class CandidateScore(BaseModel):
    candidate_index: int
    shot_alignment: int = Field(ge=1, le=5)
    character_consistency: int = Field(ge=1, le=5)
    background_continuity: int = Field(ge=1, le=5)
    prop_state_correctness: int = Field(ge=1, le=5)
    reasoning: str = ""

    @property
    def overall_score(self) -> float:
        return (
            self.shot_alignment
            + self.character_consistency
            + self.background_continuity
            + self.prop_state_correctness
        ) / 4.0


class ShotResultSummary(BaseModel):
    shot_index: int
    chosen: str
    scores: list[CandidateScore]


class Story(BaseModel):
    title: str
    shots: list[str]
    characters: list[Character] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    props: list[Prop] = Field(default_factory=list)


def make_generation_key(shot: Shot, anchors: AnchorSet, seed: int, k: int) -> str:
    payload = {
        "shot": shot.model_dump(mode="json"),
        "anchors": anchors.ordered_paths(),
        "seed": seed,
        "k": k,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
