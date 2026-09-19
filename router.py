#!/usr/bin/env python3
"""
Dual-Process AI Router: System 1 (Jev) + System 2 (Gemini)

Architecture:
- System 1 (Fast, Intuitive): Jev (TypeSafe AI) or keyword fallback
  Determines intent without text generation. 0.01ms, $0.
- System 2 (Slow, Deliberate): Gemini 3.8 Flash
  Engages only when deep reasoning, code generation, or creative writing is needed.
"""

import os
import sys
import re
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Optional: TypeSafe Jev SDK
try:
    from typesafe_sdk import TypeSafeClient, Choice, Score, Noul
    HAS_TYPESAFE = True
except ImportError:
    HAS_TYPESAFE = False

# Google Gemini SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Dynamic MCP Tool Pruning (System 1)
try:
    from tool_pruner import MCPToolPruner
    HAS_PRUNER = True
except ImportError:
    HAS_PRUNER = False



class JevClassifier:
    """
    System 1: Non-autoregressive instant classifier.
    
    Uses TypeSafe Jev API when available, falls back to
    lightweight keyword matching (still sub-millisecond).
    """

    # Customize these routing rules for your use case
    ROUTING_RULES = {
        "docker_restart": {
            "keywords": ["docker restart", "restart container",
                         "コンテナ再起動", "コンテナリスタート", "コンテナの再起動"],
            "action": "docker_restart",
            "needs_reasoning": False,
            "risk_level": "medium",
        },
        "docker_status": {
            "keywords": ["docker", "container", "コンテナ", "コンテナ一覧", "docker ps"],
            "action": "docker_status",
            "needs_reasoning": False,
            "risk_level": "low",
        },
        "git_pull": {
            "keywords": ["git pull", "プル", "リポジトリ更新", "コード最新化", "pull origin"],
            "action": "git_pull",
            "needs_reasoning": False,
            "risk_level": "medium",
        },
        "status": {
            "keywords": ["server", "status", "cpu", "memory", "disk", "uptime",
                         "サーバー", "ステータス", "メモリ", "スペック"],
            "action": "get_status",
            "needs_reasoning": False,
            "risk_level": "low",
        },
        "repos": {
            "keywords": ["github", "repo", "repository", "commit",
                         "リポジトリ", "コミット"],
            "action": "list_repos",
            "needs_reasoning": False,
            "risk_level": "low",
        },
        "files": {
            "keywords": ["drive", "file", "folder", "gdrive",
                         "ドライブ", "ファイル"],
            "action": "list_files",
            "needs_reasoning": False,
            "risk_level": "low",
        },
        "notify": {
            "keywords": ["discord", "notify", "send", "alert",
                         "通知", "送信", "連絡"],
            "action": "send_notification",
            "needs_reasoning": False,
            "risk_level": "medium",
        },
    }

    def __init__(self):
        self.client = None
        if HAS_TYPESAFE and TYPESAFE_API_KEY:
            self.client = TypeSafeClient(api_key=TYPESAFE_API_KEY)

    def classify(self, user_input: str) -> dict:
        """Classify user intent in sub-millisecond time."""
        start = time.perf_counter()

        # Try Jev API first (if available)
        if self.client:
            try:
                actions = [r["action"] for r in self.ROUTING_RULES.values()]
                actions.append("llm_reasoning")
                res = self.client.system_one(
                    state=f"User request: {user_input}",
                    questions={
                        "action": Choice(actions, "Which action best matches this request?"),
                        "confidence": Score(0, 100, "How confident are you that this request should be directly handled by this action without deep reasoning? (0=not confident, 100=absolutely certain)"),
                        "urgency": Score(0, 5, "How urgent? 0=normal, 5=critical"),
                    }
                )
                conf_score = round(res.answers.confidence.value / 100.0, 2)
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                return {
                    "source": "Jev API (TypeSafe AI)",
                    "action": res.answers.action.value,
                    "confidence": conf_score,
                    "urgency": res.answers.urgency.value,
                    "latency_ms": elapsed_ms,
                }
            except Exception as e:
                print(f"[Jev API fallback: {e}]", file=sys.stderr)

        # Keyword fallback (still 0.01ms, degraded mode)
        prompt_lower = user_input.lower()
        for rule in self.ROUTING_RULES.values():
            if any(kw in prompt_lower for kw in rule["keywords"]):
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                return {
                    "source": "Jev Classifier (keyword fallback)",
                    "action": rule["action"],
                    "confidence": 0.95,
                    "urgency": 1,
                    "latency_ms": elapsed_ms,
                }

        # Default: escalate to System 2
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "source": "Jev Classifier (keyword fallback)",
            "action": "llm_reasoning",
            "confidence": 0.20,
            "urgency": 1,
            "latency_ms": elapsed_ms,
        }


class GeminiReasoner:
    """
    System 2: Deep reasoning via Gemini 3.8 Flash.
    
    Only activated when System 1 determines that the request
    requires creative writing, complex reasoning, or code generation.
    """

    MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]

    def __init__(self, system_instruction: str = None, enable_tool_pruning: bool = True):
        self.client = genai.Client(api_key=GEMINI_API_KEY) if HAS_GENAI and GEMINI_API_KEY else None
        self.system_instruction = system_instruction or (
            "You are a helpful AI assistant. Respond concisely and accurately."
        )
        self.enable_tool_pruning = enable_tool_pruning
        self.pruner = MCPToolPruner() if HAS_PRUNER else None
        self.last_pruned_tools = []

    def generate(self, user_input: str, conversation_history: list = None, tools: list = None) -> str:
        """Generate a response using Gemini with optional conversation context and dynamic tool pruning."""
        self.last_pruned_tools = []
        if not self.client:
            return "Error: GEMINI_API_KEY is not configured."

        # Dynamically prune tools using System 1 (<0.1ms)
        genai_tools = None
        if tools and self.pruner and self.enable_tool_pruning:
            self.last_pruned_tools = self.pruner.prune(user_input, tools, top_k=3, min_score=0.15)
            if self.last_pruned_tools:
                genai_tools = self.pruner.to_genai_function_declarations(self.last_pruned_tools)
        elif tools:
            self.last_pruned_tools = tools

        # Build multi-turn context
        contents = []
        if conversation_history:
            for turn in conversation_history[-4:]:  # Keep last 4 turns
                role = "user" if turn.get("role") == "user" else "model"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=turn.get("text", ""))]
                ))
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)]
        ))

        # Try primary model, fallback to lite
        config_kwargs = {"system_instruction": self.system_instruction}
        if genai_tools:
            config_kwargs["tools"] = genai_tools

        for model_name in self.MODELS:
            try:
                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs)
                )
                return resp.text
            except Exception as e:
                print(f"[{model_name} failed: {e}, trying fallback...]", file=sys.stderr)
                time.sleep(1)

        return "LLM temporarily unavailable. Please try again."


class DualProcessRouter:
    """
    Unified orchestrator combining System 1 (Jev) + System 2 (Gemini) with Adaptive Thresholds.
    
    Usage:
        router = DualProcessRouter(use_adaptive_threshold=True)
        result = router.process("What's the server status?")
        # → Low-risk read operation, threshold 0.70, System 1 direct execution ($0)
        
        result = router.process("docker restart vaio-mcp かも")
        # → Medium-risk modification + ambiguity penalty → threshold 0.99 → escalated to System 2
    """

    RISK_THRESHOLDS = {
        "low": 0.70,       # Read-only inspection (status, docker ps, repos, files)
        "medium": 0.90,    # State-changing operations (docker restart, git pull, notify)
        "high": 0.95,      # High-risk, unknown, or destructive actions
    }

    # Indicators of hesitation, uncertainty, or ambiguity that raise the escalation threshold
    AMBIGUITY_PATTERNS = [
        r'\b(?:maybe|perhaps|probably|not sure|wondering|might)\b',
        r'(?:かも|かな|たぶん|どうだろう|かしら|っけ|じゃない|？|\?)',
    ]

    def __init__(
        self,
        tool_handlers: dict = None,
        system_instruction: str = None,
        threshold: float = 0.85,
        use_adaptive_threshold: bool = True,
        mcp_tools: list = None,
    ):
        """
        Args:
            tool_handlers: Dict mapping action names to callable functions.
            system_instruction: Custom system prompt for the Gemini reasoner.
            threshold: Default calibrated confidence threshold (0.0 - 1.0).
            use_adaptive_threshold: If True, dynamically adjusts threshold based on action risk & input ambiguity.
            mcp_tools: Optional catalog of available MCP tool definitions for dynamic pruning.
        """
        self.system1 = JevClassifier()
        self.system2 = GeminiReasoner(system_instruction=system_instruction)
        self.tool_handlers = tool_handlers or {}
        self.threshold = threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.mcp_tools = mcp_tools or []

    def get_adaptive_threshold(self, action: str, user_input: str) -> tuple[float, str]:
        """
        Calculate calibrated threshold based on action risk level and input ambiguity.

        Returns:
            (effective_threshold: float, risk_level: str)
        """
        # Determine risk level
        risk = "high"
        for rule in self.system1.ROUTING_RULES.values():
            if rule.get("action") == action:
                risk = rule.get("risk_level", "low")
                break

        base_threshold = self.RISK_THRESHOLDS.get(risk, self.threshold)

        # Ambiguity penalty: if input indicates uncertainty, raise threshold to favor System 2
        for pattern in self.AMBIGUITY_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                return min(0.99, round(base_threshold + 0.10, 2)), f"{risk} (ambiguity penalty)"

        return base_threshold, risk

    def process(self, user_input: str, conversation_history: list = None) -> dict:
        """
        Process a user request through the dual-process pipeline with adaptive calibration.
        """
        total_start = time.perf_counter()

        # --- System 1: Instant classification & calibrated confidence ---
        decision = self.system1.classify(user_input)
        action = decision["action"]
        confidence = decision.get("confidence", 0.0)
        system2_engaged = False
        output = ""

        # Determine effective threshold (Adaptive vs Static)
        if self.use_adaptive_threshold:
            effective_threshold, risk_level = self.get_adaptive_threshold(action, user_input)
        else:
            effective_threshold = self.threshold
            risk_level = "fixed"

        # Routing decision: confidence >= threshold -> System 1, else escalate
        if confidence >= effective_threshold and action != "llm_reasoning" and action in self.tool_handlers:
            # Direct tool execution (no LLM needed)
            try:
                output = self.tool_handlers[action](user_input)
            except Exception as e:
                output = f"Tool execution error: {e}"

        else:
            # --- System 2: Escalate to deep reasoning (with dynamic tool pruning) ---
            system2_engaged = True
            output = self.system2.generate(user_input, conversation_history, tools=self.mcp_tools)

        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        pruned_tool_names = [t.get("name") for t in getattr(self.system2, "last_pruned_tools", []) if isinstance(t, dict)]

        return {
            "input": user_input,
            "decision": decision,
            "confidence": confidence,
            "threshold": effective_threshold,
            "risk_level": risk_level,
            "system2_engaged": system2_engaged,
            "pruned_tools": pruned_tool_names,
            "total_latency_ms": total_ms,
            "output": output,
        }


# =============================================================================
# Demo
# =============================================================================

def _demo_get_status(_input: str) -> str:
    """Example tool handler: return mock server status."""
    return "CPU: 12%, RAM: 55% (2.0/3.7GB), Uptime: 30d, Docker: 4 containers"


def _demo_list_repos(_input: str) -> str:
    """Example tool handler: return mock repo list."""
    return "1. dual-process-ai (Private)\n2. my-website (Public)\n3. dotfiles (Private)"


def main():
    # Register your tool handlers here
    handlers = {
        "get_status": _demo_get_status,
        "list_repos": _demo_list_repos,
    }

    # Initialize router with a confidence threshold
    router = DualProcessRouter(tool_handlers=handlers, threshold=0.85)

    test_cases = [
        "What's the server status?",
        "Show my GitHub repos",
        "Design a microservice architecture for an e-commerce platform",
        "サーバーのCPU使用率は？",
    ]

    for query in test_cases:
        print(f"\n{'='*60}")
        print(f"📩 Input: {query}")
        print(f"{'='*60}")

        result = router.process(query)
        d = result["decision"]

        print(f"⚡ System 1: {d['source']}")
        print(f"   Action: {d['action']}")
        print(f"   Confidence: {d['confidence']} (Threshold: {router.threshold})")
        print(f"   Latency: {d['latency_ms']} ms")
        print(f"🧠 System 2 engaged: {'YES (escalated)' if result['system2_engaged'] else 'NO ($0)'}")
        print(f"⏱️  Total: {result['total_latency_ms']} ms")
        print(f"📤 Output: {result['output'][:200]}")


if __name__ == "__main__":
    main()
