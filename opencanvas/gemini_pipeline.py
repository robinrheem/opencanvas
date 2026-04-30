"""Gemini image-gen adapter — drop-in for diffusers' Flux2KleinPipeline.

Implements the ImagePipeline Protocol (see agents.py) by routing prompt +
reference images through Google's Gemini-3-pro-image API. Same call shape
as the local diffusers pipeline so pipeline.run treats it identically.

Use when:
    OPENCANVAS_IMAGE_BACKEND=gemini
    OPENCANVAS_IMAGE_API_KEY=<your-google-api-key>
    OPENCANVAS_IMAGE_MODEL=gemini-3-pro-image-preview
"""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import Any

from PIL import Image


class GeminiImagePipeline:
    """Adapter for Gemini's image-generation models (gemini-3-pro-image-preview,
    gemini-2.5-flash-image, etc).

    Mirrors diffusers' pipeline call shape so it slots into ImagePipeline:
      pipe(prompt=..., image=[PIL,...], num_inference_steps=..., guidance_scale=...,
           generator=...) → object with .images list

    num_inference_steps / guidance_scale are accepted but ignored — Gemini
    sets sampling internally. `generator`'s seed is also ignored (Gemini API
    does not currently expose deterministic seeding); per-candidate variation
    in K-of-K selection happens via independent API calls instead.
    """

    def __init__(self, api_key: str, model: str = "gemini-3-pro-image-preview") -> None:
        from google import genai

        if not api_key:
            raise ValueError(
                "GeminiImagePipeline requires an API key. Set "
                "OPENCANVAS_IMAGE_API_KEY (https://aistudio.google.com/apikey)."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def __call__(
        self,
        *,
        prompt: str,
        image: list[Image.Image] | None,
        num_inference_steps: int = 0,  # accepted for Protocol shape, unused
        guidance_scale: float = 0.0,   # accepted for Protocol shape, unused
        generator: Any = None,         # accepted for Protocol shape, seed unused
    ) -> SimpleNamespace:
        contents: list[Any] = [prompt]
        if image:
            contents.extend(image)  # google-genai accepts PIL.Image directly

        response = self.client.models.generate_content(
            model=self.model, contents=contents,
        )

        candidate = response.candidates[0]
        for part in candidate.content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                pil = Image.open(BytesIO(inline.data)).convert("RGB")
                return SimpleNamespace(images=[pil])
        raise RuntimeError(
            f"Gemini returned no image part for model={self.model}. "
            f"Response candidates: {response.candidates!r}"
        )

    def set_progress_bar_config(self, **_: Any) -> None:
        """No-op: API call has no local progress bar."""
        pass

    def to(self, *_a: Any, **_kw: Any) -> "GeminiImagePipeline":
        """No-op: API client doesn't move to device."""
        return self
