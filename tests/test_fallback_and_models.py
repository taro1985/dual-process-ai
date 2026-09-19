"""
Tests for System 1.5 Fallback Reasoner and Pydantic Type-Safe Data Models (Phase 1 & Phase 2)
"""
import pytest
from dpai.models import RoutingDecision, ProcessResult
from dpai.router import DualProcessRouter, System1_5FallbackReasoner


def test_models_backward_compatibility():
    """Verify RoutingDecision and ProcessResult support both object attribute and dict-like access."""
    decision = RoutingDecision(
        source="Jev Classifier (test)",
        action="get_status",
        confidence=0.95,
        urgency=1,
        latency_ms=0.03,
    )

    # Attribute access
    assert decision.action == "get_status"
    assert decision.confidence == 0.95

    # Dict index access
    assert decision["action"] == "get_status"
    assert decision["confidence"] == 0.95

    # Dict .get() access
    assert decision.get("action") == "get_status"
    assert decision.get("non_existent", "default_val") == "default_val"
    assert "action" in decision


def test_system_1_5_fallback_when_gemini_fails(monkeypatch):
    """Verify that when Gemini API fails (quota 429 or offline), System 1.5 generates helpful guidance."""
    router = DualProcessRouter(enable_habituation=False)

    # Simulate Gemini API failure (e.g. 429 RESOURCE_EXHAUSTED)
    def mock_failing_generate(*args, **kwargs):
        return "LLM temporarily unavailable. Please try again."

    monkeypatch.setattr(router.system2, "generate", mock_failing_generate)

    # Test query requesting network port inspection
    result = router.process("Show all listening network ports")

    assert isinstance(result, ProcessResult)
    assert result.is_fallback is True
    assert result.system2_engaged is True
    assert "System 1.5 Fallback Active" in result.output
    # Intent heuristics should provide actual command guidance
    assert "ss -tulpn" in result.output

    # Attribute access & Dict access both work
    assert result.output == result["output"]
    assert result.is_fallback == result["is_fallback"]


def test_system_1_5_fallback_with_principles():
    """Verify fallback reasoner leverages recalled principles directly."""
    principles = [
        "Always parse flags independently using token-based normalization.",
        "Always couple continuous daemon loops with automated benchmarks.",
    ]
    out = System1_5FallbackReasoner.fallback_reason(
        user_input="How should I parse flags safely?",
        principles=principles,
    )
    assert "System 1.5 Fallback Active" in out
    assert "Always parse flags independently" in out
