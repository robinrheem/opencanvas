"""GeminiImagePipeline adapter — call-shape compatibility with ImagePipeline."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from opencanvas.gemini_pipeline import GeminiImagePipeline


def _png_bytes(color: tuple[int, int, int] = (200, 50, 50)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (32, 32), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _fake_client(image_bytes: bytes) -> MagicMock:
    """Mock google.genai.Client returning a candidate with one inline image part."""
    inline = SimpleNamespace(data=image_bytes, mime_type="image/png")
    part = SimpleNamespace(inline_data=inline, text=None)
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    response = SimpleNamespace(candidates=[candidate])

    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def test_gemini_pipeline_rejects_missing_api_key():
    with pytest.raises(ValueError, match="OPENCANVAS_IMAGE_API_KEY"):
        GeminiImagePipeline(api_key="")


def test_gemini_pipeline_returns_pil_image():
    with patch("google.genai.Client") as mock_cls:
        mock_cls.return_value = _fake_client(_png_bytes())
        pipe = GeminiImagePipeline(api_key="dummy")
        result = pipe(
            prompt="A wide dining table",
            image=[Image.new("RGB", (16, 16))],
            num_inference_steps=0, guidance_scale=0, generator=None,
        )
    assert hasattr(result, "images")
    assert len(result.images) == 1
    assert result.images[0].size == (32, 32)
    assert result.images[0].mode == "RGB"


def test_gemini_pipeline_call_kwargs_passed_to_client():
    with patch("google.genai.Client") as mock_cls:
        client = _fake_client(_png_bytes())
        mock_cls.return_value = client
        pipe = GeminiImagePipeline(api_key="dummy", model="gemini-3-pro-image-preview")
        ref = Image.new("RGB", (8, 8))
        pipe(prompt="x", image=[ref], num_inference_steps=4, guidance_scale=1.0, generator=None)

    call = client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-3-pro-image-preview"
    contents = call.kwargs["contents"]
    assert contents[0] == "x"
    assert contents[1] is ref


def test_gemini_pipeline_raises_when_no_image_part():
    """API returned only a text part — no inline image. Pipeline must surface
    the error rather than silently returning nothing."""
    text_only_part = SimpleNamespace(inline_data=None, text="sorry, can't generate")
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[text_only_part]))
    response = SimpleNamespace(candidates=[candidate])
    client = MagicMock()
    client.models.generate_content.return_value = response

    with patch("google.genai.Client", return_value=client):
        pipe = GeminiImagePipeline(api_key="dummy")
        with pytest.raises(RuntimeError, match="no image part"):
            pipe(prompt="x", image=None, num_inference_steps=0, guidance_scale=0, generator=None)
