#!/usr/bin/env python3
"""
Dual-Process AI Safety Gate — Reflexive Guard for AI Coding Agents

A PreToolUse hook that blocks dangerous shell commands in sub-millisecond time,
BEFORE the LLM even considers executing them.

System 1 catches it. System 2 never needs to think about it.

Supported hook systems:
- Google Antigravity (PreToolUse)
- Any agent that pipes tool-call JSON to stdin

Usage (standalone test):
    echo '{"toolCall":{"name":"run_command","args":{"CommandLine":"rm -rf /"}}}' | python safety_gate.py
    # → {"decision": "deny", "reason": "⚡ [Jev Guard] Catastrophic root deletion detected."}

    echo '{"toolCall":{"name":"run_command","args":{"CommandLine":"echo hello"}}}' | python safety_gate.py
    # → {"decision": "allow"}
"""

import sys
import json
import re
import os

# Optional: TypeSafe Jev API for enhanced detection
try:
    from typesafe_sdk import TypeSafeClient, Noul, Score
    HAS_TYPESAFE = True
except ImportError:
    HAS_TYPESAFE = False


# =============================================================================
# Pattern-Based System 1 Inspector (0.01ms, zero dependencies)
# =============================================================================

# Catastrophic patterns — instant deny, no exceptions
HARD_BLOCK_PATTERNS = [
    # Root filesystem destruction
    (r'\brm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+(/\s*$|/\*|/(\s+|$))',
     "Catastrophic root deletion detected."),

    # Database destruction
    (r'\b(DROP\s+DATABASE|drop\s+database)\b',
     "DROP DATABASE detected."),

    # Filesystem formatting
    (r'\bmkfs(\.\w+)?\s+',
     "mkfs filesystem format detected."),

    # Direct block device overwrite
    (r'\bdd\s+.*of=/dev/(sd[a-z]|nvme\d+n\d+|hd[a-z]|vd[a-z])\b',
     "Direct block device overwrite detected."),

    # Secret exfiltration via pipe
    (r'(?:id_rsa|id_ed25519|/etc/shadow).*?\|\s*(?:curl|wget|nc|ncat)\b',
     "Secret key exfiltration attempt detected."),
]

# Fork bomb (literal string match)
FORK_BOMB = ':(){:|:&};:'


def inspect_command(cmd: str) -> dict:
    """
    Inspect a shell command for dangerous patterns.
    
    Returns:
        {"decision": "allow"} or {"decision": "deny", "reason": "..."}
    """
    if not cmd:
        return {"decision": "allow"}

    # 1. Literal fork bomb check
    if FORK_BOMB in cmd:
        return {"decision": "deny", "reason": "⚡ [Jev Guard] Fork bomb detected."}

    # 2. Catastrophic root deletion check (handles separated flags: rm -r -f /, rm -rf /*, rm --recursive --force /)
    if re.search(r'\brm\b', cmd):
        has_recursive = bool(re.search(r'-(?:[a-zA-Z]*r|-[a-zA-Z]*recursive)', cmd))
        has_root_target = bool(re.search(r'(?:\s+)(?:/|/\*|/\s*$|/(\s+|$))', cmd))
        if has_recursive and has_root_target:
            return {"decision": "deny", "reason": "⚡ [Jev Guard] Catastrophic root deletion detected."}

    # 3. Regex pattern matching (sub-millisecond)
    for pattern, description in HARD_BLOCK_PATTERNS:
        if re.search(pattern, cmd):
            return {"decision": "deny", "reason": f"⚡ [Jev Guard] {description}"}

    # 3. TypeSafe Jev API inspection (optional, enhanced detection)
    api_key = os.getenv("TYPESAFE_API_KEY", "")
    if HAS_TYPESAFE and api_key:
        try:
            client = TypeSafeClient(api_key=api_key)
            res = client.system_one(
                state=f"Shell command: {cmd}",
                questions={
                    "is_destructive": Noul(
                        "Is this command destructive, hostile, or likely to "
                        "permanently erase user files without consent?"
                    ),
                    "risk_level": Score(
                        0, 5,
                        "Operational risk of running this command? "
                        "(0=harmless, 5=catastrophic)"
                    ),
                }
            )
            if res.answers.is_destructive.value or res.answers.risk_level.value >= 4:
                return {
                    "decision": "deny",
                    "reason": (
                        f"⚡ [Jev System 1] Command flagged as high-risk "
                        f"(Score {res.answers.risk_level.value}/5)."
                    ),
                }
        except Exception:
            pass  # Fail-open: don't block on API errors

    return {"decision": "allow"}


# =============================================================================
# Hook Entry Point (reads JSON from stdin)
# =============================================================================

def main():
    try:
        payload_raw = sys.stdin.read()
        if not payload_raw:
            print(json.dumps({"decision": "allow"}))
            return

        payload = json.loads(payload_raw)
        tool_call = payload.get("toolCall", {})
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {})

        if tool_name == "run_command":
            cmd = args.get("CommandLine", "")
            result = inspect_command(cmd)
            print(json.dumps(result))
        else:
            print(json.dumps({"decision": "allow"}))

    except Exception:
        # Fail-open: never disrupt dev workflow on unexpected errors
        print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
