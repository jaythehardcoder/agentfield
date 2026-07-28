"""Regression test for issue #825 — Agent.run CLI dispatch."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from agentfield import Agent


def test_issue_825(monkeypatch):
    """Agent.run should dispatch supported CLI commands without starting a server."""
    cli_handler = SimpleNamespace(run_cli=Mock())
    agent = SimpleNamespace(cli_handler=cli_handler, serve=Mock())
    monkeypatch.setattr(sys, "argv", ["agent.py", "list"])

    Agent.run(agent, port=8765)

    cli_handler.run_cli.assert_called_once_with()
    agent.serve.assert_not_called()
