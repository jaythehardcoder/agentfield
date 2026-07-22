"""
Regression test for issue #586.

OpenRouter's provider matrix may reject image_config.aspect_ratio for
google/gemini-2.5-flash-image with a 404 even though the same request succeeds
without image_config. The SDK should retry without image_config for that
specific failure mode.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sdk_package_path = Path(__file__).resolve().parents[1] / "sdk" / "python" / "agentfield"
agentfield_package = types.ModuleType("agentfield")
agentfield_package.__path__ = [str(sdk_package_path)]
sys.modules.setdefault("agentfield", agentfield_package)

from agentfield.multimodal_response import MultimodalResponse
from agentfield.vision import generate_image_openrouter


def _build_openrouter_success_response():
    """Build a minimal mock OpenRouter success response."""
    mock_image_url = MagicMock()
    mock_image_url.url = "data:image/png;base64,abc"
    mock_image = MagicMock()
    mock_image.image_url = mock_image_url
    mock_choice = MagicMock()
    mock_choice.message.content = "ok"
    mock_choice.message.images = [mock_image]
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.mark.asyncio
async def test_issue_586(monkeypatch):
    """
    Reproduce OpenRouter's no-endpoints 404 for gemini-2.5-flash-image when
    image_config.aspect_ratio is sent, then verify the SDK retries without it.
    """
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    no_endpoints_404 = Exception(
        "litellm.NotFoundError: NotFoundError: OpenrouterException - "
        '{"error":{"message":"No endpoints found that support the requested '
        'output modalities: image, text","code":404}}'
    )
    success_response = _build_openrouter_success_response()

    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(
        side_effect=[
            no_endpoints_404,
            no_endpoints_404,
            no_endpoints_404,
            success_response,
        ]
    )

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        result = await generate_image_openrouter(
            prompt="A vertical portrait",
            model="openrouter/google/gemini-2.5-flash-image",
            size="1024x1024",
            quality="standard",
            style=None,
            response_format="url",
            image_config={"aspect_ratio": "9:16"},
        )

    assert isinstance(result, MultimodalResponse)
    assert result.raw_response is success_response
    assert mock_litellm.acompletion.call_count == 4

    first_call_kwargs = mock_litellm.acompletion.call_args_list[0][1]
    assert first_call_kwargs["model"] == "openrouter/google/gemini-2.5-flash-image"
    assert first_call_kwargs["modalities"] == ["image"]
    assert first_call_kwargs["image_config"] == {"aspect_ratio": "9:16"}

    retry_without_config_kwargs = mock_litellm.acompletion.call_args_list[-1][1]
    assert retry_without_config_kwargs["model"] == (
        "openrouter/google/gemini-2.5-flash-image"
    )
    assert retry_without_config_kwargs["modalities"] == ["image"]
    assert "image_config" not in retry_without_config_kwargs
