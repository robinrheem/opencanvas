from pathlib import Path

from PIL import Image

from opencanvas.agents import retrieve
from opencanvas.memory import Memory
from opencanvas.schemas import ContinuationMode, Shot


def _img(path: Path) -> Path:
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(path)
    return path


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


def test_fresh_location_returns_char_only(tmp_path: Path):
    m = Memory.empty(tmp_path / "mem")
    src = _img(tmp_path / "a.png")
    m.set_character("char-ada", "default", src)
    anchors = retrieve(_shot(0, "loc-a", ContinuationMode.fresh_location), m)

    assert len(anchors.character_refs) == 1
    assert anchors.location_ref is None
    assert anchors.previous_frame is None


def test_previous_frame_continuation_uses_prev_frame(tmp_path: Path):
    m = Memory.empty(tmp_path / "mem")
    src = _img(tmp_path / "a.png")
    m.set_character("char-ada", "default", src)
    m.set_frame(0, src)
    anchors = retrieve(_shot(1, "loc-a", ContinuationMode.previous_frame_continuation), m)

    assert anchors.previous_frame is not None
    assert anchors.location_ref is None
    assert len(anchors.character_refs) == 1


def test_location_reappearance_uses_bg_anchor(tmp_path: Path):
    m = Memory.empty(tmp_path / "mem")
    src = _img(tmp_path / "a.png")
    m.set_character("char-ada", "default", src)
    m.set_location("loc-a", src)
    anchors = retrieve(_shot(2, "loc-a", ContinuationMode.location_reappearance), m)

    assert anchors.location_ref is not None
    assert anchors.previous_frame is None
    assert len(anchors.character_refs) == 1


def test_prop_anchor_included_when_visible(tmp_path: Path):
    m = Memory.empty(tmp_path / "mem")
    src = _img(tmp_path / "a.png")
    m.set_character("char-ada", "default", src)
    m.set_prop("prop-journal", "intact", src)
    anchors = retrieve(
        _shot(
            0, "loc-a", ContinuationMode.fresh_location,
            props={"prop-journal": "intact"},
        ),
        m,
    )
    assert len(anchors.prop_refs) == 1


def test_prop_anchor_skipped_when_not_visible(tmp_path: Path):
    m = Memory.empty(tmp_path / "mem")
    src = _img(tmp_path / "a.png")
    m.set_character("char-ada", "default", src)
    m.set_prop("prop-journal", "intact", src)
    anchors = retrieve(
        _shot(
            0, "loc-a", ContinuationMode.fresh_location,
            props={"prop-journal": "not_visible"},
        ),
        m,
    )
    assert anchors.prop_refs == []


def test_canonical_anchor_used_when_no_recent(tmp_path: Path):
    m = Memory.empty(tmp_path / "mem")
    src = _img(tmp_path / "a.png")
    m.set_character_canonical("char-ada", "tuxedo", src)
    anchors = retrieve(
        _shot(
            0, "loc-a", ContinuationMode.fresh_location,
            chars={"char-ada": "tuxedo"},
        ),
        m,
    )
    assert len(anchors.character_refs) == 1
