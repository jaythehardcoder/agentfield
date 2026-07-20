"""
Regression test for issue #623: ResultCache can deadlock the Python SDK.

This reproduces the deadlock path where AgentFieldClient._await_execution_async
enters ResultCache from the event-loop thread while another thread owns the
cache's threading lock. A blocked threading lock acquisition freezes the loop,
so asyncio.wait_for cannot fire.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import sys
import threading
import types
from pathlib import Path

import pytest

SDK_ROOT = Path(__file__).resolve().parents[1] / "sdk" / "python"
sys.path.insert(0, str(SDK_ROOT))

agentfield_pkg = types.ModuleType("agentfield")
agentfield_pkg.__path__ = [str(SDK_ROOT / "agentfield")]
sys.modules.setdefault("agentfield", agentfield_pkg)

from agentfield.async_config import AsyncConfig
from agentfield.client import AgentFieldClient, _Submission


class DummyResponse:
    def __init__(self, payload: dict):
        self.status_code = 200
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _run_deadlock_probe() -> None:
    client = AgentFieldClient(
        base_url="http://example.test",
        api_key="api-key",
        async_config=AsyncConfig(
            initial_poll_interval=0.01,
            max_poll_interval=0.01,
            max_execution_timeout=30.0,
        ),
    )
    submission = _Submission(
        execution_id="exec-issue-623",
        run_id="run-issue-623",
        target="node.reasoner",
        status="queued",
    )

    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_result_cache_lock() -> None:
        with client._result_cache._lock:
            lock_acquired.set()
            release_lock.wait()

    holder = threading.Thread(target=hold_result_cache_lock)
    holder.start()
    assert lock_acquired.wait(timeout=1.0)

    async def fake_request(method, url, **kwargs):
        return DummyResponse({"status": "running"})

    async def run_waiter() -> None:
        client._async_request = fake_request
        try:
            await asyncio.wait_for(
                client._await_execution_async(
                    submission,
                    {"X-Run-ID": submission.run_id},
                ),
                timeout=0.2,
            )
        except asyncio.TimeoutError:
            pass
        finally:
            release_lock.set()

    try:
        asyncio.run(run_waiter())
    finally:
        release_lock.set()
        holder.join(timeout=1.0)


def test_issue_623():
    process = multiprocessing.Process(target=_run_deadlock_probe)
    process.start()
    process.join(timeout=2.0)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)

    if process.exitcode != 0:
        pytest.fail(
            "AgentFieldClient._await_execution_async deadlocked while reading ResultCache"
        )
