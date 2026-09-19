"""
Tests for HabituationEngine & S2 -> S1 Reflex Compilation
"""
import pytest
import time
from dpai.habituation import HabituationEngine
from dpai.router import DualProcessRouter


def test_habituation_engine_basic(tmp_path):
    """Verify basic habituate and match lifecycle."""
    cache_file = tmp_path / "habits.json"
    engine = HabituationEngine(persistence_path=str(cache_file))

    # Initial state
    assert engine.match("PostgreSQLの起動コマンド") is None

    # Learn a habit
    habit = engine.habituate(
        query="PostgreSQLの起動コマンドを教えてください",
        response="sudo systemctl start postgresql",
        action="habituated_response",
        confidence=0.96,
    )
    assert habit["hit_count"] == 1
    assert cache_file.exists()

    # Match exact
    res1 = engine.match("PostgreSQLの起動コマンドを教えてください")
    assert res1 is not None
    assert res1["response"] == "sudo systemctl start postgresql"
    assert res1["hit_count"] >= 2

    # Match similar query (fuzzy via token/stemming)
    res2 = engine.match("PostgreSQLの起動コマンド教えて")
    assert res2 is not None
    assert res2["response"] == "sudo systemctl start postgresql"


def test_habituation_persistence(tmp_path):
    """Verify saving and loading habits across engine restarts."""
    cache_file = tmp_path / "habits.json"
    engine1 = HabituationEngine(persistence_path=str(cache_file))
    engine1.habituate("docker restart command", "docker restart <container_id>")

    # Restart engine with same file
    engine2 = HabituationEngine(persistence_path=str(cache_file))
    res = engine2.match("docker restart command")
    assert res is not None
    assert res["response"] == "docker restart <container_id>"


def test_router_habituation_workflow(tmp_path):
    """Verify DualProcessRouter uses S1 habituated reflexes to bypass LLM on repeated queries."""
    cache_file = tmp_path / "router_habits.json"
    router = DualProcessRouter(
        enable_habituation=True,
        habit_persistence_path=str(cache_file),
    )

    # 1. Teach or manually habituate a reflex
    router.habituate(
        query="How do I check system temperature?",
        response="Use 'sensors' command to inspect CPU temperature.",
    )

    # 2. Query matching habit: System 1 MUST serve it directly without LLM
    result = router.process("How do I check system temperature?")
    assert result["is_habituated"] is True
    assert result["system2_engaged"] is False
    assert result["total_latency_ms"] < 20.0  # sub-millisecond reflex
    assert "sensors" in result["output"]

    # 3. Similar phrasing should also hit habit
    result_similar = router.process("Check system temperature")
    assert result_similar["is_habituated"] is True
    assert result_similar["system2_engaged"] is False
    assert "sensors" in result_similar["output"]


def test_router_auto_habituation_from_system2(monkeypatch, tmp_path):
    """Verify that successful System 2 resolution automatically habituates into System 1."""
    cache_file = tmp_path / "auto_habits.json"
    router = DualProcessRouter(
        enable_habituation=True,
        habit_persistence_path=str(cache_file),
    )

    # Mock System 2 response
    def mock_generate(*args, **kwargs):
        return "To list listening ports, execute: sudo ss -tulpn"

    monkeypatch.setattr(router.system2, "generate", mock_generate)

    # Turn 1: Unseen query triggers System 2
    query = "How to inspect listening network ports?"
    res1 = router.process(query)
    assert res1["system2_engaged"] is True
    assert res1["is_habituated"] is False
    assert "sudo ss -tulpn" in res1["output"]

    # Turn 2: Exact or similar query MUST NOW be handled by System 1 Reflex!
    res2 = router.process(query)
    assert res2["is_habituated"] is True
    assert res2["system2_engaged"] is False  # LLM completely bypassed!
    assert "sudo ss -tulpn" in res2["output"]
