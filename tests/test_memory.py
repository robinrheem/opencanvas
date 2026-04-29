from pathlib import Path

from opencanvas.memory import Memory, safe_filename


def test_memory_roundtrip(tmp_path: Path, make_image):
    src = make_image(name="src.png", color=(1, 2, 3), size=4)
    m = Memory.empty(tmp_path / "mem")
    m.add_character("char-ada", "default", src)
    m.add_location("loc-warehouse", src)
    m.add_frame(0, src)
    m.save()

    loaded = Memory.load_or_empty(tmp_path / "mem")
    assert loaded.character("char-ada", "default") is not None
    assert loaded.character("char-ada", "default").read_bytes() == src.read_bytes()
    assert loaded.locations.get("loc-warehouse") is not None
    assert loaded.frames.get(0) is not None


def test_empty_memory(tmp_path: Path):
    m = Memory.empty(tmp_path / "mem")
    assert m.character("x", "y") is None
    assert m.locations.get("x") is None
    assert m.frames.get(0) is None


def test_safe_filename_collapses_unsafe_chars():
    assert safe_filename("3/4_full") == "3_4_full"
    assert safe_filename("on the table") == "on_the_table"
    assert safe_filename("wine-glass:b") == "wine-glass_b"
    assert safe_filename("///") == "_"  # all-stripped fallback


def test_has_predicates(tmp_path: Path, make_image):
    src = make_image(name="x.png", size=4)
    m = Memory.empty(tmp_path / "mem")
    assert not m.has_character("c1", "default")
    assert not m.has_prop("p1", "default")
    assert not m.has_location("loc-a")
    m.add_character("c1", "default", src)
    m.add_prop("p1", "default", src)
    m.add_location("loc-a", src)
    assert m.has_character("c1", "default")
    assert not m.has_character("c1", "tuxedo")  # different state = different key
    assert m.has_prop("p1", "default")
    assert m.has_location("loc-a")


def test_add_prop_with_unsafe_state(tmp_path: Path, make_image):
    """Regression: LLM emits state names like '3/4_full'; must not break path joins."""
    src = make_image(name="src.png", size=4)
    m = Memory.empty(tmp_path / "mem")
    out = m.add_prop("prop-wine-glass-b", "3/4_full", src)
    assert out.exists()
    assert "/" not in out.name  # filename component clean
    # Logical lookup uses original (unsanitized) state string.
    assert m.props[("prop-wine-glass-b", "3/4_full")] == out
