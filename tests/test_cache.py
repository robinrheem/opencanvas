"""Cache + canonical-reference + memory-resume tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from opencanvas.config import Settings
from opencanvas.pipeline import run
from opencanvas.schemas import Story


def test_second_run_hits_cache(tmp_path: Path, patched_llm, torch_stub, fake_pipeline):
    """On resume, persisted memory enriches shot 0's anchors, so its cache key
    differs and shot 0 regenerates. Shots 1..T-1 still hit cache because their
    anchor paths are stable across runs."""
    story = Story.model_validate_json(Path("examples/mini_story.json").read_bytes())
    settings = Settings(
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        k_candidates=2,
    )

    run(story, settings, image_pipeline=fake_pipeline)
    first_calls = fake_pipeline.call_count
    assert first_calls == settings.k_candidates * len(story.shots)

    fake_pipeline.calls.clear()
    run(story, settings, image_pipeline=fake_pipeline)
    # Only shot 0 misses → at most K calls.
    assert fake_pipeline.call_count <= settings.k_candidates


def test_canonical_reference_seeded(tmp_path: Path, patched_llm, torch_stub, fake_pipeline):
    ref = tmp_path / "ada_canonical.png"
    Image.new("RGB", (16, 16), color=(200, 100, 50)).save(ref)
    story = Story.model_validate(
        {
            "title": "tiny",
            "shots": ["Ada at the door."],
            "characters": [
                {
                    "id": "char-ada",
                    "name": "Ada",
                    "description": "engineer",
                    "reference_image": str(ref),
                }
            ],
        }
    )
    settings = Settings(
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        k_candidates=1,
    )
    run(story, settings, image_pipeline=fake_pipeline)

    canonical = settings.out_dir / "memory" / "characters_canonical" / "char-ada__default.png"
    assert canonical.exists()
    assert canonical.read_bytes() == ref.read_bytes()


def test_memory_persists_across_runs(tmp_path: Path, patched_llm, torch_stub, fake_pipeline):
    """Resume: second run loads memory.manifest.json instead of starting empty."""
    story = Story.model_validate_json(Path("examples/mini_story.json").read_bytes())
    settings = Settings(
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        k_candidates=1,
    )

    run(story, settings, image_pipeline=fake_pipeline)
    manifest_first = (settings.out_dir / "memory" / "manifest.json").read_text()
    frames_dir = settings.out_dir / "memory" / "frames"
    frame_count_first = len(list(frames_dir.iterdir()))

    fake_pipeline.calls.clear()
    run(story, settings, image_pipeline=fake_pipeline)
    manifest_second = (settings.out_dir / "memory" / "manifest.json").read_text()

    assert manifest_second == manifest_first
    assert len(list(frames_dir.iterdir())) == frame_count_first
