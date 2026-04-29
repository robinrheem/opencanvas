"""Tests for bbox-based anchor crops + memory any-state fallback."""

from pathlib import Path

from PIL import Image

from opencanvas.agents import (
    _format_prop_states_with_carriers,
    crop_to_anchor,
    extract_location_anchor,
    segment_to_anchor,
)
from opencanvas.memory import Memory
from opencanvas.schemas import BBox


def _half_opaque_red(crop, session=None, post_process_mask=False):
    """Centre 50% opaque red, rest transparent."""
    w, h = crop.size
    out = Image.new("RGBA", (w, h), (255, 0, 0, 0))
    for x in range(w // 4, (3 * w) // 4):
        for y in range(h // 4, (3 * h) // 4):
            out.putpixel((x, y), (255, 0, 0, 255))
    return out


def _all_opaque_red(crop, session=None, post_process_mask=False):
    """Whole region opaque foreground."""
    return Image.new("RGBA", crop.size, (255, 0, 0, 255))


def _must_not_call(*a, **kw):
    raise AssertionError("rembg.remove should not be called")


def test_crop_to_anchor_resizes(make_image):
    src = make_image(size=100)
    dest = src.parent / "char_crop.png"

    crop_to_anchor(src, BBox(x=0.25, y=0.25, w=0.5, h=0.5), dest)

    with Image.open(dest) as out:
        assert out.size == (50, 50)


def test_crop_to_anchor_clamps_bbox(make_image):
    src = make_image(size=100)
    dest = src.parent / "edge_crop.png"

    crop_to_anchor(src, BBox(x=0.9, y=0.9, w=0.5, h=0.5), dest)

    with Image.open(dest) as out:
        # Right edge clamped to 100, top from 90 → 10x10
        assert out.size == (10, 10)


def test_crop_to_anchor_pixel_coords_autodetected(make_image):
    """Regression: VLM emits bbox as pixel coords (e.g. x=12, w=311).
    Without auto-detect, multiply-by-frame-size blows past the frame and
    both ends clamp to width — bbox returns the full image, silently
    breaking subject anchoring."""
    src = make_image(size=400)
    dest = src.parent / "px_crop.png"

    crop_to_anchor(src, BBox(x=12.0, y=43.0, w=311.0, h=300.0), dest)

    with Image.open(dest) as out:
        assert out.size == (311, 300)


def test_crop_to_anchor_degenerate_falls_back_to_full(make_image):
    src = make_image(color=(50, 50, 50))
    dest = src.parent / "fallback.png"

    crop_to_anchor(src, BBox(x=1.0, y=1.0, w=0.01, h=0.01), dest)

    with Image.open(dest) as out:
        assert out.size == (100, 100)


def test_prop_any_state_returns_most_recent(tmp_path: Path, make_image):
    src = make_image(name="src.png", size=4)
    m = Memory.empty(tmp_path / "mem")
    m.add_prop("prop-journal", "intact", src)
    m.add_prop("prop-journal", "burned", src)
    m.add_prop("other-prop", "default", src)

    found = m.prop_any_state("prop-journal")
    assert found is not None
    # Most recent insertion order = "burned"
    assert "burned" in str(found)


def test_prop_any_state_unknown_returns_none(tmp_path: Path):
    m = Memory.empty(tmp_path / "mem")
    assert m.prop_any_state("nonexistent") is None


def test_format_prop_states_with_carriers():
    s = _format_prop_states_with_carriers(
        {"prop-journal": "carried", "prop-key": "intact"},
        {"prop-journal": "char-ada"},
    )
    assert "prop-journal=carried (carried by char-ada)" in s
    # prop-key has no carrier — must NOT show "(carried by ...)"
    after_key = s.split("prop-key=intact")[1]
    assert "(carried by" not in after_key


def test_format_prop_states_empty():
    assert _format_prop_states_with_carriers({}, {}) == "(none)"


def test_segment_to_anchor_composites_subject_on_neutral_bg(make_image, mock_rembg):
    mock_rembg(_half_opaque_red)
    src = make_image(color=(0, 0, 0))
    dest = src.parent / "anchor.png"

    segment_to_anchor(
        src, BBox(x=0.0, y=0.0, w=1.0, h=1.0), dest,
        model_name="ignored", bg_color=(50, 60, 70),
    )

    with Image.open(dest) as out:
        # Top-left was transparent in the mock → reveals neutral bg
        assert out.getpixel((1, 1)) == (50, 60, 70)
        # Centre was opaque red → preserved
        assert out.getpixel((50, 50)) == (255, 0, 0)


def test_extract_location_anchor_inpaints_subject_bboxes(make_image, mock_rembg, monkeypatch):
    """Inpainting path: subject silhouettes replaced via Big-LaMa inpaint, no
    silhouette-shaped gaps left."""
    import opencanvas.agents as agents_mod

    mock_rembg(_all_opaque_red)

    def fake_lama_inpaint(rgb_image, mask, device="cuda"):
        # Pretend inpainter recolored the masked area to bright blue.
        out = rgb_image.copy()
        blue = Image.new("RGB", out.size, (0, 0, 255))
        out.paste(blue, mask=mask)
        return out

    monkeypatch.setattr(agents_mod, "_lama_inpaint", fake_lama_inpaint)

    src = make_image(color=(200, 200, 200))
    dest = src.parent / "loc.png"

    extract_location_anchor(
        src, [BBox(x=0.0, y=0.0, w=0.5, h=0.5)], dest,
        model_name="ignored", enable_inpainting=True,
    )

    with Image.open(dest) as out:
        # Inside bbox: silhouette inpainted (fake = blue)
        assert out.getpixel((10, 10)) == (0, 0, 255)
        # Outside bbox: original frame preserved
        assert out.getpixel((90, 90)) == (200, 200, 200)


def test_extract_location_anchor_neutral_plate_fallback(make_image, mock_rembg):
    """Fallback path (enable_inpainting=False) keeps the paper-faithful
    neutral-gray silhouette behavior."""
    mock_rembg(_all_opaque_red)
    src = make_image(color=(200, 200, 200))
    dest = src.parent / "loc.png"

    extract_location_anchor(
        src, [BBox(x=0.0, y=0.0, w=0.5, h=0.5)], dest,
        model_name="ignored", bg_color=(50, 60, 70), enable_inpainting=False,
    )

    with Image.open(dest) as out:
        assert out.getpixel((10, 10)) == (50, 60, 70)
        assert out.getpixel((90, 90)) == (200, 200, 200)


def test_extract_location_anchor_no_subjects_returns_full_frame(make_image, mock_rembg):
    mock_rembg(_must_not_call)
    src = make_image(color=(10, 20, 30))
    dest = src.parent / "loc.png"

    extract_location_anchor(src, [], dest, model_name="ignored")

    with Image.open(dest) as out:
        assert out.size == (100, 100)
        assert out.getpixel((50, 50)) == (10, 20, 30)


def test_segment_to_anchor_degenerate_bbox_falls_back(make_image, mock_rembg):
    """Degenerate bbox short-circuits before rembg is touched."""
    mock_rembg(_must_not_call)
    src = make_image(color=(10, 20, 30))
    dest = src.parent / "anchor.png"

    segment_to_anchor(src, BBox(x=1.0, y=1.0, w=0.01, h=0.01), dest, model_name="ignored")

    with Image.open(dest) as out:
        assert out.size == (100, 100)
