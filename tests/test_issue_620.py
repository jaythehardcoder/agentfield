"""
Regression test for issue #620: async reasoner execution from an async caller.

FastAPI and similar ASGI hosts already run inside an event loop. The Python SDK's
serverless/decorator path must not call ``asyncio.run`` inside that loop when it
dispatches an async reasoner.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk" / "python"))


class _StubOKPKey:
    @classmethod
    def generate_key(cls, *_args, **_kwargs):
        return cls()

    def as_dict(self, *_args, **_kwargs):
        return {}


sys.modules.setdefault("joserfc", types.SimpleNamespace(jwe=types.SimpleNamespace()))
sys.modules.setdefault("joserfc.jwk", types.SimpleNamespace(OKPKey=_StubOKPKey))

from agentfield.agent import Agent


def _make_agent(monkeypatch: pytest.MonkeyPatch) -> Agent:
    """Construct a minimal Agent without network side effects."""
    monkeypatch.setattr("agentfield.agent._detect_container_ip", lambda: None)
    monkeypatch.setattr("agentfield.agent._detect_local_ip", lambda: "127.0.0.1")
    monkeypatch.setattr("agentfield.agent._is_running_in_container", lambda: False)

    return Agent(
        node_id="issue-620-agent",
        agentfield_server="http://localhost:8080",
        dev_mode=True,
        callback_url="http://localhost:8001",
        auto_register=False,
    )


@pytest.mark.asyncio
async def test_issue_620(monkeypatch):
    """
    Calling a decorated async reasoner through the serverless dispatcher from an
    already-running event loop should not fail with nested ``asyncio.run``.
    """
    agent = _make_agent(monkeypatch)

    @agent.reasoner()
    async def async_echo(message: str) -> dict[str, str]:
        return {"message": message}

    result = agent.handle_serverless(
        {
            "reasoner": "async_echo",
            "input": {"message": "hello from the event loop"},
        }
    )

    assert result["statusCode"] == 200, result
    assert result["body"] == {"message": "hello from the event loop"}
