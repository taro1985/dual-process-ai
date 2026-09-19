import pytest
from router import JevClassifier, DualProcessRouter


def test_jev_classifier_routine_status():
    classifier = JevClassifier()
    res = classifier.classify("What is the server status?")
    assert res["action"] == "get_status"
    assert res["confidence"] >= 0.85
    assert res["latency_ms"] < 50.0  # sub-millisecond in degraded mode


def test_jev_classifier_routine_repos():
    classifier = JevClassifier()
    res = classifier.classify("Show my github repos")
    assert res["action"] == "list_repos"
    assert res["confidence"] >= 0.85


def test_jev_classifier_escalation_reasoning():
    classifier = JevClassifier()
    res = classifier.classify("Design a complex distributed consensus algorithm for our cluster")
    assert res["action"] == "llm_reasoning"
    assert res["confidence"] < 0.85


def test_dual_process_router_fast_path():
    tool_called = False

    def mock_status(_input: str) -> str:
        nonlocal tool_called
        tool_called = True
        return "mock_status_ok"

    router = DualProcessRouter(
        tool_handlers={"get_status": mock_status},
        threshold=0.85,
    )

    result = router.process("Show server status")
    assert tool_called is True
    assert result["system2_engaged"] is False
    assert result["output"] == "mock_status_ok"
    assert result["confidence"] >= 0.85
    assert result["total_latency_ms"] < 100.0
