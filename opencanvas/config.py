from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

CACHE_TAG_GENERATE = "generate"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENCANVAS_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "gemma4:31b"

    image_model: str = "black-forest-labs/FLUX.2-klein-4B"
    device: str = "cuda"  # e.g. "cuda", "cuda:0", "cuda:1", "cpu"
    k_candidates: int = 4
    num_inference_steps: int = 4   # FLUX.2 [klein] is step-distilled (~4 steps)
    guidance_scale: float = 1.0    # FLUX.2 [klein] is guidance-distilled (CFG ignored)
    seed: int = 42
    seed_stride: int = 1000

    cache_dir: Path = Path("./cache")
    out_dir: Path = Path("./out")

    # Anchor extraction (Algorithm 4)
    segment_model: str = "birefnet-general"
    segment_bg_color: tuple[int, int, int] = (128, 128, 128)
    enable_segmentation: bool = True
    # Per-run observability: JSONL of every LLM/VLM call, per-shot decision
    # summaries, cache hit/miss audit. Disable for benchmarks where the disk
    # cost matters.
    enable_logging: bool = True
    # Big-LaMa inpainting for location anchors. When False, falls back to the
    # neutral-gray silhouette (paper-faithful but model reads silhouettes as
    # "fill with people" → forced cast count). Inpainting closes the holes
    # with coherent surrounding pixels so the location anchor is a clean
    # empty scene with no compositional pressure on the image generator.
    enable_inpainting: bool = True

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
