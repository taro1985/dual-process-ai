"""
DPAI (Dual-Process AI)
~~~~~~~~~~~~~~~~~~~~~~

A high-performance Python framework combining System 1 (sub-millisecond reflexive
classification and safety verification) with System 2 (deep reasoning with Gemini).

Inspired by Daniel Kahneman's "Thinking, Fast and Slow".
"""

from .router import DualProcessRouter, JevClassifier, GeminiReasoner
from .safety_gate import inspect_command
from .hermes_gate import HermesSafetyGate, check_command_safe
from .tool_pruner import MCPToolPruner
from .memory_scorer import JevMemoryScorer, parse_evolved_principles, load_evolved_principles
from .habituation import HabituationEngine
from .models import RoutingDecision, ProcessResult
from .persistent_memory import SQLiteEpisodicMemory
from .binary_protocol import BinaryActionPacket, BitwiseLatentMatcher

__version__ = "0.2.0"
__all__ = [
    "DualProcessRouter",
    "JevClassifier",
    "GeminiReasoner",
    "HermesSafetyGate",
    "inspect_command",
    "check_command_safe",
    "MCPToolPruner",
    "JevMemoryScorer",
    "parse_evolved_principles",
    "load_evolved_principles",
    "HabituationEngine",
    "RoutingDecision",
    "ProcessResult",
    "SQLiteEpisodicMemory",
    "BinaryActionPacket",
    "BitwiseLatentMatcher",
]
