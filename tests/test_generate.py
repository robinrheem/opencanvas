from pathlib import Path

from opencanvas.agents import generate
from opencanvas.config import Settings
from opencanvas.schemas import AnchorSet, ContinuationMode, Shot


def _shot() -> Shot:
    return Shot(
        index=2,
        description="warehouse wide shot",
        location_id="loc-a",
        continuation_mode=ContinuationMode.fresh_location,
    )


def test_generate_writes_k_candidates(tmp_path: Path, torch_stub, fake_pipeline, tmp_image):
    settings = Settings(out_dir=tmp_path / "out", k_candidates=3)
    anchors = AnchorSet(character_refs=[str(tmp_image)])

    paths = generate(_shot(), anchors, settings, fake_pipeline, seed=1000)

    assert len(paths) == 3
    assert all(p.exists() for p in paths)
    assert fake_pipeline.call_count == 3
    assert [c["generator"].initial_seed() for c in fake_pipeline.calls] == [1000, 1001, 1002]
    assert fake_pipeline.calls[0]["true_cfg_scale"] == 4.0
    assert fake_pipeline.calls[0]["guidance_scale"] == 1.0
    assert isinstance(fake_pipeline.calls[0]["image"], list)


def test_generate_with_no_anchors_uses_gray(tmp_path: Path, torch_stub, fake_pipeline):
    settings = Settings(out_dir=tmp_path / "out", k_candidates=1)
    shot = Shot(
        index=0, description="cold open",
        location_id="loc-a", continuation_mode=ContinuationMode.fresh_location,
    )

    paths = generate(shot, AnchorSet(), settings, fake_pipeline, seed=7)

    assert len(paths) == 1
    assert len(fake_pipeline.calls[0]["image"]) == 1
