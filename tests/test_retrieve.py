from pathlib import Path

from opencanvas.agents import retrieve
from opencanvas.memory import Memory
from opencanvas.schemas import (
    Character,
    ContinuationMode,
    Location,
    Plan,
    Shot,
)


def _make_plan(shots: list[Shot]) -> Plan:
    return Plan(
        shots=shots,
        characters=[Character(id="char-ada", name="Ada", description="")],
        locations=[Location(id=s.location_id, name=s.location_id, description="") for s in shots if s.location_id],
    )


def _shot(
    index: int,
    location_id: str,
    mode: ContinuationMode,
    *,
    chars: dict[str, str] | None = None,
    props: dict[str, str] | None = None,
) -> Shot:
    return Shot(
        index=index,
        description=f"s{index}",
        location_id=location_id,
        continuation_mode=mode,
        character_states=chars or {"char-ada": "default"},
        prop_states=props or {},
    )


def test_fresh_location_returns_char_only(tmp_path: Path, make_image):
    m = Memory.empty(tmp_path / "mem")
    src = make_image(name="a.png", color=(1, 2, 3), size=4)
    m.add_character("char-ada", "default", src)
    shot = _shot(0, "loc-a", ContinuationMode.fresh_location)
    anchors = retrieve(shot, _make_plan([shot]), m)

    assert len(anchors.character_refs) == 1
    assert anchors.location_ref is None
    assert anchors.previous_frame is None


def test_previous_frame_continuation_uses_prev_frame(tmp_path: Path, make_image):
    m = Memory.empty(tmp_path / "mem")
    src = make_image(name="a.png", color=(1, 2, 3), size=4)
    m.add_character("char-ada", "default", src)
    m.add_frame(0, src)
    s0 = _shot(0, "loc-a", ContinuationMode.fresh_location)
    s1 = _shot(1, "loc-a", ContinuationMode.previous_frame_continuation)
    anchors = retrieve(s1, _make_plan([s0, s1]), m)

    assert anchors.previous_frame is not None
    assert anchors.location_ref is None
    assert len(anchors.character_refs) == 1


def test_location_reappearance_uses_bg_anchor(tmp_path: Path, make_image):
    m = Memory.empty(tmp_path / "mem")
    src = make_image(name="a.png", color=(1, 2, 3), size=4)
    m.add_character("char-ada", "default", src)
    m.add_location("loc-a", src)
    shots = [
        _shot(0, "loc-a", ContinuationMode.fresh_location),
        _shot(1, "loc-b", ContinuationMode.fresh_location),
        _shot(2, "loc-a", ContinuationMode.location_reappearance),
    ]
    anchors = retrieve(shots[2], _make_plan(shots), m)

    assert anchors.location_ref is not None
    assert anchors.previous_frame is None
    assert len(anchors.character_refs) == 1


def test_prop_anchor_included_when_visible(tmp_path: Path, make_image):
    m = Memory.empty(tmp_path / "mem")
    src = make_image(name="a.png", color=(1, 2, 3), size=4)
    m.add_character("char-ada", "default", src)
    m.add_prop("prop-journal", "intact", src)
    shot = _shot(0, "loc-a", ContinuationMode.fresh_location, props={"prop-journal": "intact"})
    anchors = retrieve(shot, _make_plan([shot]), m)
    assert len(anchors.prop_refs) == 1


def test_prop_anchor_skipped_when_not_visible(tmp_path: Path, make_image):
    m = Memory.empty(tmp_path / "mem")
    src = make_image(name="a.png", color=(1, 2, 3), size=4)
    m.add_character("char-ada", "default", src)
    m.add_prop("prop-journal", "intact", src)
    shot = _shot(0, "loc-a", ContinuationMode.fresh_location, props={"prop-journal": "not_visible"})
    anchors = retrieve(shot, _make_plan([shot]), m)
    assert anchors.prop_refs == []


def test_canonical_used_when_appearance_changes(tmp_path: Path, make_image):
    """Algorithm 2: appearance_state changed across shots → prefer canonical anchor."""
    m = Memory.empty(tmp_path / "mem")
    src = make_image(name="tuxedo.png", color=(1, 2, 3), size=4)
    m.add_canonical("char-ada", "tuxedo", src)
    m.add_character("char-ada", "tuxedo", src)

    s0 = _shot(0, "loc-a", ContinuationMode.fresh_location, chars={"char-ada": "default"})
    s1 = _shot(1, "loc-a", ContinuationMode.previous_frame_continuation, chars={"char-ada": "tuxedo"})
    anchors = retrieve(s1, _make_plan([s0, s1]), m)

    # appearance changed default -> tuxedo, so canonical wins
    assert "characters_canonical" in anchors.character_refs[0]


def test_recent_used_when_state_unchanged(tmp_path: Path, make_image):
    m = Memory.empty(tmp_path / "mem")
    src = make_image(name="tuxedo.png", color=(1, 2, 3), size=4)
    m.add_canonical("char-ada", "tuxedo", src)
    m.add_character("char-ada", "tuxedo", src)

    s0 = _shot(0, "loc-a", ContinuationMode.fresh_location, chars={"char-ada": "tuxedo"})
    s1 = _shot(1, "loc-a", ContinuationMode.previous_frame_continuation, chars={"char-ada": "tuxedo"})
    anchors = retrieve(s1, _make_plan([s0, s1]), m)

    # appearance unchanged tuxedo -> tuxedo, so recent wins (NOT canonical)
    assert "characters_canonical" not in anchors.character_refs[0]
    assert "characters/char-ada__tuxedo" in anchors.character_refs[0]
