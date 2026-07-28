"""Regression test for issue #624 — Agent startup must not probe IP metadata.

On Kubernetes, metadata-service addresses are commonly protected by a
NetworkPolicy.  An explicitly configured callback URL makes external IP
discovery unnecessary, so creating an Agent must not request cloud metadata
endpoints or api.ipify.org.
"""

import pytest

from agentfield import Agent


def test_issue_624(monkeypatch):
    """Creating an Agent with an explicit callback URL must avoid IP probes."""
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.96.0.1")
    monkeypatch.setattr("agentfield.agent._detect_local_ip", lambda: "127.0.0.1")

    def unexpected_request(url, **kwargs):
        pytest.fail(f"Agent startup unexpectedly probed {url}")

    monkeypatch.setattr("requests.get", unexpected_request)

    agent = Agent(
        node_id="issue-624-agent",
        agentfield_server="http://localhost:8080",
        callback_url="http://issue-624-agent:8000",
        auto_register=False,
        enable_did=False,
    )

    assert agent.base_url == "http://issue-624-agent:8000"
