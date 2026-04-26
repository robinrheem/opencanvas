"""Tests for bbox-based anchor crops + memory any-state fallback."""

from pathlib import Path

from PIL import Image

from opencanvas.agents import (
    _format_prop_states_with_carriers,
    crop_to_anchor,
)
from opencanvas.memory import Memory
from opencanvas.schemas import BBox


def test_crop_to_anchor_resizes(tmp_path: Path):
    src = tmp_path / "frame.png"
    Image.new("RGB", (100, 100), color=(0, 0, 0)).save(src)
    dest = tmp_path / "char_crop.png"

    crop_to_anchor(src, BBox(x=0.25, y=0.25, w=0.5, h=0.5), dest)

    with Image.open(dest) as out:
        assert out.size == (50, 50)


def test_crop_to_anchor_clamps_bbox(tmp_path: Path):
    src = tmp_path / "frame.png"
    Image.new("RGB", (100, 100), color=(0, 0, 0)).save(src)
    dest = tmp_path / "edge_crop.png"

    crop_to_anchor(src, BBox(x=0.9, y=0.9, w=0.5, h=0.5), dest)

    with Image.open(dest) as out:
        # Right edge clamped to 100, top from 90 → 10x10
        assert out.size == (10, 10)


def test_crop_to_anchor_degenerate_falls_back_to_full(tmp_path: Path):
    src = tmp_path / "frame.png"
    Image.new("RGB", (100, 100), color=(50, 50, 50)).save(src)
    dest = tmp_path / "fallback.png"

    crop_to_anchor(src, BBox(x=1.0, y=1.0, w=0.01, h=0.01), dest)

    with Image.open(dest) as out:
        assert out.size == (100, 100)


def test_get_prop_any_state_returns_most_recent(tmp_path: Path):
    src = tmp_path / "src.png"
    Image.new("RGB", (4, 4)).save(src)
    m = Memory.empty(tmp_path / "mem")
    m.set_prop("prop-journal", "intact", src)
    m.set_prop("prop-journal", "burned", src)
    m.set_prop("other-prop", "default", src)

    found = m.get_prop_any_state("prop-journal")
    assert found is not None
    # Most recent insertion order = "burned"
    assert "burned" in str(found)


def test_get_prop_any_state_unknown_returns_none(tmp_path: Path):
    m = Memory.empty(tmp_path / "mem")
    assert m.get_prop_any_state("nonexistent") is None


def test_format_prop_states_with_carriers():
    s = _format_prop_states_with_carriers(
        {"prop-journal": "carried", "prop-key": "intact"},
        {"prop-journal": "char-ada"},
    )
    assert "prop-journal=carried (carried by char-ada)" in s
    assert "prop-key=intact" in s
    assert "(carried by" not in s.split("prop-key=intact")[1] if "prop-key=intact" in s else True


def test_format_prop_states_empty():
    assert _format_prop_states_with_carriers({}, {}) == "(none)"


def test_segment_to_anchor_composites_subject_on_neutral_bg(tmp_path: Path, monkeypatch):
    """rembg returns RGBA with alpha mask; segment_to_anchor composites onto bg_color."""
    import opencanvas.agents as agents_mod
    from opencanvas.agents import segment_to_anchor

    def _fake_remove(crop, session=None, post_process_mask=False):
        # Pretend rembg kept the centre 50% as foreground, made the rest transparent.
        w, h = crop.size
        out = Image.new("RGBA", (w, h), (255, 0, 0, 0))  # transparent red
        cx0, cy0 = w // 4, h // 4
        cx1, cy1 = (3 * w) // 4, (3 * h) // 4
        for x in range(cx0, cx1):
            for y in range(cy0, cy1):
                out.putpixel((x, y), (255, 0, 0, 255))  # opaque red
        return out

    monkeypatch.setattr(agents_mod, "_rembg_session", lambda model_name: None)
    monkeypatch.setattr("rembg.remove", _fake_remove)

    src = tmp_path / "frame.png"
    Image.new("RGB", (100, 100), color=(0, 0, 0)).save(src)
    dest = tmp_path / "anchor.png"

    segment_to_anchor(
        src,
        BBox(x=0.0, y=0.0, w=1.0, h=1.0),
        dest,
        model_name="ignored",
        bg_color=(50, 60, 70),
    )

    with Image.open(dest) as out:
        # Top-left pixel was transparent in the mock → reveals neutral bg
        assert out.getpixel((1, 1)) == (50, 60, 70)
        # Centre pixel was opaque red → preserved
        assert out.getpixel((50, 50)) == (255, 0, 0)


def test_segment_to_anchor_degenerate_bbox_falls_back(tmp_path: Path, monkeypatch):
    """Degenerate bbox short-circuits before rembg is touched."""
    import opencanvas.agents as agents_mod
    from opencanvas.agents import segment_to_anchor

    called = {"n": 0}

    def _fake_remove(*a, **kw):
        called["n"] += 1
        raise RuntimeError("must not be called")

    monkeypatch.setattr(agents_mod, "_rembg_session", lambda model_name: None)
    monkeypatch.setattr("rembg.remove", _fake_remove)

    src = tmp_path / "frame.png"
    Image.new("RGB", (100, 100), color=(10, 20, 30)).save(src)
    dest = tmp_path / "anchor.png"

    segment_to_anchor(
        src, BBox(x=1.0, y=1.0, w=0.01, h=0.01), dest, model_name="ignored"
    )

    assert called["n"] == 0
    with Image.open(dest) as out:
        assert out.size == (100, 100)
