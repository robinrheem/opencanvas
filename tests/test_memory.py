from pathlib import Path

from PIL import Image

from opencanvas.memory import Memory


def _img(path: Path) -> Path:
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(path)
    return path


def test_memory_roundtrip(tmp_path: Path):
    src = _img(tmp_path / "src.png")
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
