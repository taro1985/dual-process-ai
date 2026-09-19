"""
Tests for SQLite-backed Persistent Episodic Memory
"""
import pytest
from dpai.persistent_memory import SQLiteEpisodicMemory


def test_persistent_memory_lifecycle(tmp_path):
    """Verify session turns creation, retrieval, and pruning."""
    db_file = tmp_path / "test_memory.sqlite"
    mem = SQLiteEpisodicMemory(db_path=str(db_file))

    session_id = "test-channel-123"

    # Add turns
    mem.append_turn(session_id, "user", "What is the server CPU load?", max_turns_per_session=3)
    mem.append_turn(session_id, "model", "CPU load is 12%.", max_turns_per_session=3)
    mem.append_turn(session_id, "user", "And memory usage?", max_turns_per_session=3)

    turns = mem.get_turns(session_id)
    assert len(turns) == 3
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == "What is the server CPU load?"
    assert turns[2]["content"] == "And memory usage?"

    # Add 4th turn to trigger pruning (max=3)
    mem.append_turn(session_id, "model", "Memory usage is 1.8GB / 4GB.", max_turns_per_session=3)
    turns_after = mem.get_turns(session_id)
    assert len(turns_after) == 3
    # The oldest turn should be pruned
    assert turns_after[0]["content"] == "CPU load is 12%."
    assert turns_after[2]["content"] == "Memory usage is 1.8GB / 4GB."


def test_persistent_memory_persistence_across_restarts(tmp_path):
    """Verify data survives instance recreation (simulating bot restart)."""
    db_file = tmp_path / "test_restart.sqlite"

    mem1 = SQLiteEpisodicMemory(db_path=str(db_file))
    mem1.append_turn("channel-A", "user", "Turn from session before restart")

    # Recreate instance with same db path
    mem2 = SQLiteEpisodicMemory(db_path=str(db_file))
    turns = mem2.get_turns("channel-A")
    assert len(turns) == 1
    assert turns[0]["content"] == "Turn from session before restart"
