import pytest
from safety_gate import inspect_command


def test_safety_gate_blocks_root_deletion():
    dangerous_cmds = [
        "rm -rf /",
        "rm -rf /*",
        "rm -r -f /",
        "rm -rf /tmp/.. /",
    ]
    for cmd in dangerous_cmds:
        res = inspect_command(cmd)
        assert res["decision"] == "deny", f"Failed to deny: {cmd}"
        assert "Catastrophic" in res["reason"] or "root" in res["reason"]


def test_safety_gate_blocks_drop_database():
    res = inspect_command("psql -c 'DROP DATABASE production;'")
    assert res["decision"] == "deny"
    assert "DROP DATABASE" in res["reason"]


def test_safety_gate_blocks_fork_bomb():
    res = inspect_command(":(){:|:&};:")
    assert res["decision"] == "deny"
    assert "Fork bomb" in res["reason"]


def test_safety_gate_blocks_mkfs():
    res = inspect_command("mkfs.ext4 /dev/sda1")
    assert res["decision"] == "deny"
    assert "mkfs" in res["reason"]


def test_safety_gate_allows_safe_commands():
    safe_cmds = [
        "echo hello world",
        "git status",
        "python router.py",
        "ls -la /var/log",
        "cat README.md",
    ]
    for cmd in safe_cmds:
        res = inspect_command(cmd)
        assert res["decision"] == "allow", f"False positive on safe command: {cmd}"
