from pathlib import Path

from opencanvas.agents import generate
from opencanvas.config import Settings
from opencanvas.schemas import AnchorSet, BackgroundPlan, ContinuationMode, Shot


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
    bg = BackgroundPlan(shot_index=2, must_appear=["prop-journal"])

    paths = generate(_shot(), anchors, bg, settings, fake_pipeline, seed=1000)

    assert len(paths) == 3
    assert all(p.exists() for p in paths)
    assert fake_pipeline.call_count == 3
    assert [c["generator"].initial_seed() for c in fake_pipeline.calls] == [1000, 1001, 1002]
    assert fake_pipeline.calls[0]["guidance_scale"] == 1.0
    assert fake_pipeline.calls[0]["num_inference_steps"] == 4
    assert isinstance(fake_pipeline.calls[0]["image"], list)
    assert "true_cfg_scale" not in fake_pipeline.calls[0]   # FLUX.2 doesn't take this
    assert "negative_prompt" not in fake_pipeline.calls[0]   # nor this
    assert "must_appear" in fake_pipeline.calls[0]["prompt"]


def test_generate_pads_refs_to_square(tmp_path: Path, torch_stub, fake_pipeline):
    """Regression: portrait char crops + square location → output portrait.
    All refs get padded to square at >=64px to neutralize aspect-bias."""
    from PIL import Image

    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (32, 200), color=(0, 0, 0)).save(tiny)
    settings = Settings(out_dir=tmp_path / "out", k_candidates=1)
    anchors = AnchorSet(character_refs=[str(tiny)])

    paths = generate(_shot(), anchors, None, settings, fake_pipeline, seed=1)

    assert len(paths) == 1
    ref = fake_pipeline.calls[0]["image"][0]
    w, h = ref.size
    assert w == h
    assert w >= 64


def test_generate_with_no_anchors_uses_gray(tmp_path: Path, torch_stub, fake_pipeline):
    settings = Settings(out_dir=tmp_path / "out", k_candidates=1)
    shot = Shot(
        index=0, description="cold open",
        location_id="loc-a", continuation_mode=ContinuationMode.fresh_location,
    )

    paths = generate(shot, AnchorSet(), None, settings, fake_pipeline, seed=7)

    assert len(paths) == 1
    assert len(fake_pipeline.calls[0]["image"]) == 1
