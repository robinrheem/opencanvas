from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

import opencanvas.agents as agents_mod


class _FakeResult:
    def __init__(self, imgs):
        self.images = imgs


class FakeImagePipeline:
    """Records call kwargs; returns one image colored from the seed."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        seed = kwargs["generator"].initial_seed()
        return _FakeResult([Image.new("RGB", (8, 8), color=(seed % 256, 0, 0))])

    @property
    def call_count(self) -> int:
        return len(self.calls)


class _TorchStub:
    class Generator:
        def __init__(self, device: str = "cpu"):
            self._seed = 0

        def manual_seed(self, s: int) -> "_TorchStub.Generator":
            self._seed = s
            return self

        def initial_seed(self) -> int:
            return self._seed


@pytest.fixture
def torch_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _TorchStub)
    return _TorchStub


@pytest.fixture
def fake_pipeline() -> FakeImagePipeline:
    return FakeImagePipeline()


@pytest.fixture
def patched_llm(monkeypatch):
    def _factory(output_type, instructions, settings):
        return Agent(TestModel(), output_type=output_type, instructions=instructions)

    monkeypatch.setattr(agents_mod, "_agent_for", _factory)


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    p = tmp_path / "img.png"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(p)
    return p


@pytest.fixture
def make_image(tmp_path: Path) -> Callable[..., Path]:
    """Factory: writes a PNG into tmp_path and returns its path."""
    def _make(name: str = "frame.png", color: tuple = (0, 0, 0), size: int = 100) -> Path:
        p = tmp_path / name
        Image.new("RGB", (size, size), color=color).save(p)
        return p
    return _make


@pytest.fixture
def mock_rembg(monkeypatch) -> Callable[[Callable], None]:
    """Factory: install a fake `rembg.remove` and stub `_rembg_session`."""
    def _install(fake_remove: Callable) -> None:
        monkeypatch.setattr(agents_mod, "_rembg_session", lambda model_name: None)
        monkeypatch.setattr("rembg.remove", fake_remove)
    return _install
