"""refine_prev_frame_anchor: handle cast diff under previous_frame_continuation."""

from pathlib import Path

from PIL import Image

from opencanvas.agents import refine_prev_frame_anchor
from opencanvas.config import Settings
from opencanvas.memory import Memory
from opencanvas.schemas import (
    AnchorSet,
    BBox,
    Character,
    CharacterState,
    ContinuationMode,
    Location,
    Plan,
    Shot,
)


def _shot(idx: int, mode: ContinuationMode, *, chars: dict[str, str]) -> Shot:
    return Shot(
        index=idx, description=f"s{idx}", location_id="loc-a",
        continuation_mode=mode, character_states=chars,
    )


def _plan(shots: list[Shot]) -> Plan:
    return Plan(
        shots=shots,
        characters=[Character(id="c1", name="c1", description=""),
                    Character(id="c2", name="c2", description="")],
        locations=[Location(id="loc-a", name="A", description="")],
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(out_dir=tmp_path / "out", cache_dir=tmp_path / "cache")


def test_no_change_passthrough_when_cast_unchanged(tmp_path: Path, make_image):
    m = Memory.empty(tmp_path / "mem")
    src = make_image(name="prev.png")
    m.add_frame(0, src)
    s0 = _shot(0, ContinuationMode.fresh_location, chars={"c1": "default"})
    s1 = _shot(1, ContinuationMode.previous_frame_continuation, chars={"c1": "default"})
    anchors = AnchorSet(previous_frame=str(m.frames[0]))

    out = refine_prev_frame_anchor(s1, _plan([s0, s1]), anchors, m, _settings(tmp_path),
                                   masked_dir=tmp_path / "masked")

    assert out.previous_frame == str(m.frames[0])  # untouched


def test_arrival_demotes_to_location_reappearance(tmp_path: Path, make_image):
    m = Memory.empty(tmp_path / "mem")
    src = make_image(name="prev.png")
    m.add_frame(0, src)
    m.add_location("loc-a", src)
    s0 = _shot(0, ContinuationMode.fresh_location,
               chars={"c1": "default", "c2": CharacterState.not_present})
    s1 = _shot(1, ContinuationMode.previous_frame_continuation,
               chars={"c1": "default", "c2": "default"})  # c2 arrives
    anchors = AnchorSet(previous_frame=str(m.frames[0]))

    out = refine_prev_frame_anchor(s1, _plan([s0, s1]), anchors, m, _settings(tmp_path),
                                   masked_dir=tmp_path / "masked")

    assert out.previous_frame is None
    assert out.location_ref == str(m.locations["loc-a"])


def test_departure_masks_prev_frame(tmp_path: Path, make_image, mock_rembg):
    m = Memory.empty(tmp_path / "mem")
    prev = make_image(name="prev.png", color=(50, 60, 70), size=64)
    m.add_frame(0, prev)
    m.set_frame_bboxes(0, {"c2": BBox(x=0.5, y=0.0, w=0.5, h=1.0)})

    def fake_remove(img, **kwargs):
        # Pretend the right half is fully foreground (alpha=255), left fully background.
        return Image.new("RGBA", img.size, (0, 0, 0, 255))

    mock_rembg(fake_remove)

    s0 = _shot(0, ContinuationMode.fresh_location,
               chars={"c1": "default", "c2": "default"})
    s1 = _shot(1, ContinuationMode.previous_frame_continuation,
               chars={"c1": "default", "c2": CharacterState.not_present})  # c2 departs
    anchors = AnchorSet(previous_frame=str(m.frames[0]))

    out = refine_prev_frame_anchor(s1, _plan([s0, s1]), anchors, m, _settings(tmp_path),
                                   masked_dir=tmp_path / "masked")

    assert out.previous_frame is not None
    assert out.previous_frame != str(m.frames[0])  # rewritten
    assert (tmp_path / "masked" / "shot_0001.png").exists()


def test_departure_without_stored_bbox_passthrough(tmp_path: Path, make_image):
    """No stored bbox for the departing char → cannot mask, leave prev_frame intact."""
    m = Memory.empty(tmp_path / "mem")
    src = make_image(name="prev.png")
    m.add_frame(0, src)
    # Note: NO set_frame_bboxes call.
    s0 = _shot(0, ContinuationMode.fresh_location,
               chars={"c1": "default", "c2": "default"})
    s1 = _shot(1, ContinuationMode.previous_frame_continuation,
               chars={"c1": "default", "c2": CharacterState.not_present})
    anchors = AnchorSet(previous_frame=str(m.frames[0]))

    out = refine_prev_frame_anchor(s1, _plan([s0, s1]), anchors, m, _settings(tmp_path),
                                   masked_dir=tmp_path / "masked")

    assert out.previous_frame == str(m.frames[0])  # untouched


def test_non_prev_frame_mode_passthrough(tmp_path: Path, make_image):
    m = Memory.empty(tmp_path / "mem")
    src = make_image(name="prev.png")
    m.add_frame(0, src)
    s0 = _shot(0, ContinuationMode.fresh_location, chars={"c1": "default"})
    s1 = _shot(1, ContinuationMode.fresh_location, chars={"c1": "default"})
    anchors = AnchorSet()

    out = refine_prev_frame_anchor(s1, _plan([s0, s1]), anchors, m, _settings(tmp_path),
                                   masked_dir=tmp_path / "masked")

    assert out is anchors  # no-op
