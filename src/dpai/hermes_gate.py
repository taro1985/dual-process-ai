#!/usr/bin/env python3
"""
Hermes Safety Gate — Reflexive Command Inspection Hook for HermesAgent

Integrates Dual-Process AI's System 1 (Jev / regex inspector) into HermesAgent's
execution loop, blocking destructive or hostile commands before execution in <0.1ms.
"""

import sys
from pathlib import Path
from typing import Tuple

try:
    from .safety_gate import inspect_command
except ImportError:
    from safety_gate import inspect_command



class HermesSafetyGate:
    """
    Sub-millisecond safety gate for autonomous agents like HermesAgent.
    Acts as System 1 reflexive defense before any tool or shell invocation.
    """

    @staticmethod
    def inspect(command: str) -> Tuple[bool, str]:
        """
        Inspect a shell command for dangerous patterns.

        Returns:
            (is_allowed: bool, reason: str)
            If allowed: (True, "")
            If denied: (False, "⚡ [Jev Guard] Reason...")
        """
        if not command:
            return True, ""

        result = inspect_command(command)
        if result.get("decision") == "deny":
            reason = result.get("reason", "⚡ [Jev Guard] Execution denied by safety policy.")
            return False, reason
        return True, ""


def check_command_safe(command: str) -> Tuple[bool, str]:
    """Convenience functional interface."""
    return HermesSafetyGate.inspect(command)
