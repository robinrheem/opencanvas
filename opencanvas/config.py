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

    image_model: str = "Qwen/Qwen-Image-Edit-2509"
    k_candidates: int = 4
    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    seed: int = 42
    seed_stride: int = 1000

    cache_dir: Path = Path("./cache")
    out_dir: Path = Path("./out")

    # Anchor extraction (Algorithm 4)
    segment_model: str = "birefnet-general"
    segment_bg_color: tuple[int, int, int] = (128, 128, 128)
    enable_segmentation: bool = True

    def ensure_dirs(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
