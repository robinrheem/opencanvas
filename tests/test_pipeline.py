"""End-to-end pipeline test with TestModel + fake image pipeline."""

from __future__ import annotations

from pathlib import Path

from opencanvas.config import Settings
from opencanvas.pipeline import run
from opencanvas.schemas import Story


def test_pipeline_end_to_end(tmp_path: Path, patched_llm, torch_stub, fake_pipeline):
    story = Story.model_validate_json(Path("examples/mini_story.json").read_bytes())
    settings = Settings(
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        k_candidates=2,
    )

    plan_result, shot_results = run(story, settings, image_pipeline=fake_pipeline)

    assert (settings.out_dir / "plan.json").exists()
    assert (settings.out_dir / "results.json").exists()
    assert len(shot_results) == len(story.shots)
    for r in shot_results:
        assert len(r.candidates) == settings.k_candidates
        assert all(p.exists() for p in r.candidates)
        assert r.chosen.exists()
    assert (settings.out_dir / "memory" / "manifest.json").exists()
    assert any((settings.out_dir / "memory" / "frames").iterdir())
