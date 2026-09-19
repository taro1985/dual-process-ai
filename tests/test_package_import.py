"""
Test importing and using DPAI as an installed package (from dpai import ...).
"""
import pytest
import dpai
from dpai import (
    DualProcessRouter,
    JevClassifier,
    GeminiReasoner,
    HermesSafetyGate,
    inspect_command,
    check_command_safe,
    MCPToolPruner,
    JevMemoryScorer,
    __version__,
)


def test_package_version_and_exports():
    """Verify package metadata and all exported symbols."""
    assert __version__ == "0.2.0"
    assert dpai.__version__ == "0.2.0"
    assert DualProcessRouter is not None
    assert JevClassifier is not None
    assert GeminiReasoner is not None
    assert HermesSafetyGate is not None
    assert inspect_command is not None
    assert check_command_safe is not None
    assert MCPToolPruner is not None
    assert JevMemoryScorer is not None
    assert dpai.HabituationEngine is not None
    assert dpai.RoutingDecision is not None
    assert dpai.ProcessResult is not None
    assert dpai.SQLiteEpisodicMemory is not None
    assert dpai.BinaryActionPacket is not None
    assert dpai.BitwiseLatentMatcher is not None


def test_package_safety_gate():
    """Verify safety gate directly from dpai package."""
    result = inspect_command("drop database prod")
    assert result["decision"] == "deny"

    allowed, reason = check_command_safe("ls -la")
    assert allowed is True
    assert reason == ""


def test_package_router_system1():
    """Verify router fast path works when initialized via package."""
    router = DualProcessRouter(threshold=0.8)
    decision = router.classify("Show my GitHub repos")
    assert decision["action"] == "list_repos"
    decision2 = router.classifier.classify("Show my GitHub repos")
    assert decision2["action"] == "list_repos"


def test_package_tool_pruner():
    """Verify MCPToolPruner works via package."""
    pruner = MCPToolPruner()
    catalog = [
        {"name": "git_commit", "description": "Commit changes"},
        {"name": "weather_api", "description": "Get weather"},
    ]
    pruned = pruner.prune("git push", catalog, top_k=1)
    assert len(pruned) == 1
    assert pruned[0]["name"] == "git_commit"


def test_package_memory_scorer():
    """Verify JevMemoryScorer works via package."""
    scorer = JevMemoryScorer()
    episodes = [
        {"id": 1, "task": "docker restart container", "outcome": "success"},
        {"id": 2, "task": "cook a meal", "outcome": "success"},
    ]
    top = scorer.rank_and_filter("restart docker", episodes, threshold=0.1, limit=1)
    assert len(top) == 1
    assert top[0]["id"] == 1

