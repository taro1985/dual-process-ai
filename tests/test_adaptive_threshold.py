#!/usr/bin/env python3
"""
Tests for Adaptive Threshold (Risk-aware Routing & Ambiguity Penalty).
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from router import DualProcessRouter


def test_adaptive_threshold_by_risk_level():
    router = DualProcessRouter(use_adaptive_threshold=True)

    # Low risk (read-only inspection)
    t_status, r_status = router.get_adaptive_threshold("get_status", "server status")
    assert t_status == 0.70
    assert r_status == "low"

    t_docker_status, r_docker_status = router.get_adaptive_threshold("docker_status", "docker ps")
    assert t_docker_status == 0.70
    assert r_docker_status == "low"

    # Medium risk (state-changing operations)
    t_restart, r_restart = router.get_adaptive_threshold("docker_restart", "docker restart app")
    assert t_restart == 0.90
    assert r_restart == "medium"

    t_pull, r_pull = router.get_adaptive_threshold("git_pull", "git pull")
    assert t_pull == 0.90
    assert r_pull == "medium"


def test_adaptive_threshold_ambiguity_penalty():
    router = DualProcessRouter(use_adaptive_threshold=True)

    # Clear instruction: standard base threshold
    t_clear, _ = router.get_adaptive_threshold("docker_restart", "docker restart vaio-mcp")
    assert t_clear == 0.90

    # Ambiguous / uncertain inputs: triggers +0.10 penalty
    ambiguous_prompts = [
        "docker restart vaio-mcp かも",
        "docker restart vaio-mcp じゃない？",
        "docker restart vaio-mcp どうだろう",
        "maybe docker restart vaio-mcp",
        "docker restart vaio-mcp?",
    ]
    for p in ambiguous_prompts:
        t_amb, r_amb = router.get_adaptive_threshold("docker_restart", p)
        assert t_amb == 0.99 or t_amb == 1.0, f"Prompt '{p}' should have ambiguity penalty, got {t_amb}"
        assert "ambiguity penalty" in r_amb


def test_adaptive_threshold_routing_escalation():
    # Mock handler
    executed = []
    handlers = {
        "docker_restart": lambda inp: executed.append(inp) or "restarted",
        "get_status": lambda inp: executed.append(inp) or "healthy",
    }
    router = DualProcessRouter(tool_handlers=handlers, use_adaptive_threshold=True)

    # 1. Clear status request (low risk) -> executes via System 1
    res1 = router.process("server status")
    assert not res1["system2_engaged"]
    assert "healthy" in res1["output"]
    assert res1["threshold"] == 0.70

    # 2. Ambiguous restart request -> escalated to System 2 due to ambiguity penalty
    res2 = router.process("docker restart vaio-mcp かも？")
    assert res2["system2_engaged"]
    assert res2["threshold"] >= 0.99
    assert "ambiguity penalty" in res2["risk_level"]
