#!/usr/bin/env python3
"""
Tests for Remote Tools routing and safety (Docker & Git pull).
"""

import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from router import JevClassifier, DualProcessRouter
from discord_bot import restart_docker_container, TOOL_HANDLERS


def test_jev_classifier_routes_docker_and_git():
    classifier = JevClassifier()

    r1 = classifier.classify("docker ps")
    assert r1["action"] == "docker_status"
    assert r1["confidence"] >= 0.85

    r2 = classifier.classify("コンテナ一覧見せて")
    assert r2["action"] == "docker_status"

    r3 = classifier.classify("docker restart vaio-mcp")
    assert r3["action"] == "docker_restart"

    r4 = classifier.classify("git pull origin main")
    assert r4["action"] == "git_pull"

    r5 = classifier.classify("コード最新化して")
    assert r5["action"] == "git_pull"


def test_docker_restart_blocks_injection():
    # Attempting command injection via container name
    malicious = [
        "docker restart myapp; rm -rf /",
        "docker restart vaio && cat /etc/shadow",
        "docker restart container`id`",
        "docker restart $(whoami)",
    ]
    for cmd in malicious:
        res_raw = restart_docker_container(cmd)
        res = json.loads(res_raw)
        assert res.get("status") == "error"
        assert "不正なコンテナ名" in res.get("error") or "⚡ [Jev Guard]" in res.get("error")


def test_dual_process_router_fast_path_remote():
    # Mock handlers
    mock_handlers = {
        "docker_status": lambda inp: json.dumps({"status": "success", "containers": [{"names": "mcp-server", "status": "Up"}]}),
        "git_pull": lambda inp: json.dumps({"status": "success", "output": "Already up to date."}),
    }
    router = DualProcessRouter(tool_handlers=mock_handlers)

    res_docker = router.process("docker status")
    assert not res_docker["system2_engaged"]
    assert "mcp-server" in res_docker["output"]

    res_git = router.process("git pull")
    assert not res_git["system2_engaged"]
    assert "Already up to date." in res_git["output"]
