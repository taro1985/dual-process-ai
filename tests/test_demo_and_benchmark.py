#!/usr/bin/env python3
"""
Tests for demo_cli and benchmark scripts.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from router import DualProcessRouter
from tool_pruner import MCPToolPruner
from demo_cli import run_sample_query, MOCK_CATALOG
from benchmark import run_benchmark


def test_demo_cli_sample_queries(capsys):
    handlers = {
        "get_status": lambda inp: '{"status": "ok"}',
        "docker_status": lambda inp: '{"containers": []}',
    }
    router = DualProcessRouter(tool_handlers=handlers, use_adaptive_threshold=True)
    # Mock LLM to prevent external API latency in tests
    router.system2.generate = lambda *a, **k: "Mocked reasoning output"
    pruner = MCPToolPruner()


    # 1. Test routine query (System 1)
    run_sample_query(router, pruner, "server status")
    captured = capsys.readouterr()
    assert "[System 1 / Jev Fast Path] DIRECT EXECUTION" in captured.out
    assert "0.000000" in captured.out

    # 2. Test dangerous command (Safety Gate)
    run_sample_query(router, pruner, "rm -rf /")
    captured_safety = capsys.readouterr()
    assert "[System 1 Safety Gate] EXECUTION BLOCKED" in captured_safety.out

    # 3. Test reasoning escalation
    run_sample_query(router, pruner, "Design a microservice architecture")
    captured_reason = capsys.readouterr()
    assert "[System 2 / Gemini Reasoning] ESCALATED" in captured_reason.out
    assert "MCP Dynamic Pruning" in captured_reason.out


def test_benchmark_run(capsys):
    # Run a small benchmark simulation
    run_benchmark(iterations_per_item=2)
    captured = capsys.readouterr()
    assert "DPAI vs Traditional All-to-LLM Architecture Benchmark" in captured.out
    assert "Efficiency Gains" in captured.out
    assert "Cost Reduction" in captured.out
