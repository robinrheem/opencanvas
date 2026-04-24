from pathlib import Path

from PIL import Image

from opencanvas.memory import Memory


def _img(path: Path) -> Path:
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(path)
    return path


def test_memory_roundtrip(tmp_path: Path):
    src = _img(tmp_path / "src.png")
    m = Memory.empty(tmp_path / "mem")
    m.set_character("char-ada", "default", src)
    m.set_location("loc-warehouse", src)
    m.set_frame(0, src)
    m.save()

    loaded = Memory.load_or_empty(tmp_path / "mem")
    assert loaded.get_character("char-ada", "default") is not None
    assert loaded.get_character("char-ada", "default").read_bytes() == src.read_bytes()
    assert loaded.get_location("loc-warehouse") is not None
    assert loaded.get_frame(0) is not None


def test_empty_memory(tmp_path: Path):
    m = Memory.empty(tmp_path / "mem")
    assert m.get_character("x", "y") is None
    assert m.get_location("x") is None
    assert m.get_frame(0) is None
