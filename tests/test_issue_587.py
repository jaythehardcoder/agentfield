"""Regression test for data URLs returned by Gemini image models."""

import base64

from agentfield.multimodal_response import ImageOutput


def test_issue_587(tmp_path, monkeypatch):
    image_bytes = b"gemini-image-bytes"
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
    output_path = tmp_path / "out.png"

    def fail_if_requested(*args, **kwargs):
        raise AssertionError("data URLs must be decoded locally, not fetched")

    monkeypatch.setattr("requests.get", fail_if_requested)

    ImageOutput(url=data_url).save(output_path)

    assert output_path.read_bytes() == image_bytes
