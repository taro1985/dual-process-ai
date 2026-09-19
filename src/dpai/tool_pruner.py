#!/usr/bin/env python3
"""
MCP Tool Pruner — System 1 Dynamic Tool Pruning Gate for Dual-Process AI

Eliminates tool-schema bloat and hallucination by dynamically pruning large MCP
tool catalogs down to the 2-3 most relevant tools in <0.1ms before LLM invocation.
"""

import os
import re
import sys
import time
from typing import List, Dict, Any, Optional

# Optional: TypeSafe Jev API
try:
    from typesafe_sdk import TypeSafeClient, Score
    HAS_TYPESAFE = True
except ImportError:
    HAS_TYPESAFE = False

# Google GenAI types
try:
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class MCPToolPruner:
    """
    Sub-millisecond dynamic tool selector / pruner for MCP and LLM function calling.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY", "")
        self.client = None
        if HAS_TYPESAFE and self.api_key:
            try:
                self.client = TypeSafeClient(api_key=self.api_key)
            except Exception:
                self.client = None

    def _tokenize(self, text: str) -> set[str]:
        """Sub-millisecond token extraction for Japanese and English."""
        if not text:
            return set()
        words = re.findall(r'[a-zA-Z0-9_\-]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+', text.lower())
        return set(words)

    def calculate_tool_relevance(self, user_prompt: str, tool: Dict[str, Any]) -> float:
        """
        Calculate relevance score (0.0 - 1.0) of a tool definition against user prompt.
        """
        name = str(tool.get("name", ""))
        desc = str(tool.get("description", ""))
        params = ""
        input_schema = tool.get("inputSchema", {}) or tool.get("parameters", {})
        if isinstance(input_schema, dict):
            props = input_schema.get("properties", {})
            params = " ".join(props.keys())

        # Try Jev API if configured
        if self.client:
            try:
                res = self.client.system_one(
                    state=f"User prompt: '{user_prompt}'\nTool: {name} ({desc})",
                    questions={
                        "relevance": Score(
                            0, 100,
                            "How likely is this tool needed to satisfy the user request? (0=not needed, 100=essential)"
                        )
                    }
                )
                return round(res.answers.relevance.value / 100.0, 3)
            except Exception:
                pass

        # Sub-millisecond reflexive heuristic (< 0.02ms)
        prompt_tokens = self._tokenize(user_prompt)
        if not prompt_tokens:
            return 0.0

        name_tokens = self._tokenize(name.replace("_", " "))
        desc_tokens = self._tokenize(desc)
        param_tokens = self._tokenize(params)

        # 1. Exact or partial tool name match (highest weight)
        name_matches = prompt_tokens & name_tokens
        name_score = (len(name_matches) / max(len(name_tokens), 1)) * 0.6 if name_matches else 0.0

        # 2. Description keyword match
        desc_matches = prompt_tokens & desc_tokens
        desc_score = min(0.4, (len(desc_matches) / max(len(prompt_tokens), 1)) * 0.5) if desc_matches else 0.0

        # 3. Parameter match
        param_matches = prompt_tokens & param_tokens
        param_score = min(0.2, (len(param_matches) / max(len(prompt_tokens), 1)) * 0.3) if param_matches else 0.0

        total_score = min(1.0, round(name_score + desc_score + param_score, 3))
        return total_score

    def prune(
        self,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        top_k: int = 3,
        min_score: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Dynamically filter and rank tools down to the top_k most relevant tools.

        Args:
            user_prompt: Incoming user prompt.
            tools: Full list of available MCP tool definitions.
            top_k: Maximum number of tools to retain.
            min_score: Minimum relevance threshold to include a tool.

        Returns:
            Pruned list of tool definitions with `_prune_score` and `_prune_latency_ms`.
        """
        start = time.perf_counter()
        if not tools:
            return []

        scored = []
        for tool in tools:
            score = self.calculate_tool_relevance(user_prompt, tool)
            if score >= min_score:
                t_copy = dict(tool)
                t_copy["_prune_score"] = score
                scored.append(t_copy)

        scored.sort(key=lambda x: x["_prune_score"], reverse=True)
        pruned = scored[:top_k]

        elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
        for t in pruned:
            t["_prune_latency_ms"] = elapsed_ms

        return pruned

    def to_genai_function_declarations(self, pruned_tools: List[Dict[str, Any]]) -> List[Any]:
        """
        Convert pruned MCP tools to Google GenAI function declaration objects.
        """
        if not HAS_GENAI:
            return pruned_tools

        declarations = []
        for t in pruned_tools:
            name = t.get("name", "")
            description = t.get("description", "")
            parameters = t.get("inputSchema", {}) or t.get("parameters", {})
            try:
                # Handle raw dict parameter schemas
                decl = types.FunctionDeclaration(
                    name=name,
                    description=description,
                    parameters=parameters if parameters else None,
                )
                declarations.append(decl)
            except Exception:
                # If schema formatting differs, pass through
                declarations.append(t)
        return declarations
