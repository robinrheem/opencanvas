from opencanvas.schemas import (
    AnchorSet,
    CandidateScore,
    Character,
    ContinuationMode,
    Plan,
    Shot,
    Story,
)


def test_plan_roundtrip():
    p = Plan(
        shots=[
            Shot(
                index=0,
                description="intro",
                location_id="loc-a",
                continuation_mode=ContinuationMode.fresh_location,
                character_states={"char-ada": "default"},
            )
        ],
        characters=[Character(id="char-ada", name="Ada", description="engineer")],
        locations=[],
    )
    dumped = p.model_dump_json()
    restored = Plan.model_validate_json(dumped)
    assert restored == p


def test_candidate_score_overall():
    s = CandidateScore(
        candidate_index=0,
        shot_alignment=5,
        character_consistency=4,
        background_continuity=3,
        prop_state_correctness=2,
    )
    assert s.overall_score == 3.5


def test_anchorset_ordering():
    a = AnchorSet(
        character_refs=["/c.png"],
        location_ref="/bg.png",
        previous_frame="/prev.png",
        prop_refs=["/p.png"],
    )
    # Order: prev frame, characters, location, props
    assert a.ordered_paths() == ["/prev.png", "/c.png", "/bg.png", "/p.png"]


def test_story_parsing():
    raw = {"title": "Demo", "shots": ["shot a", "shot b"]}
    story = Story.model_validate(raw)
    assert len(story.shots) == 2
    assert story.characters == []


def test_continuation_mode_string():
    m = ContinuationMode.previous_frame_continuation
    assert str(m) == "previous_frame_continuation"
