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


class JevClassifier:
    """
    System 1: Non-autoregressive instant classifier.
    
    Uses TypeSafe Jev API when available, falls back to
    lightweight keyword matching (still sub-millisecond).
    """

    # Customize these routing rules for your use case
    ROUTING_RULES = {
        "status": {
            "keywords": ["server", "status", "cpu", "memory", "disk", "uptime",
                         "サーバー", "ステータス", "メモリ", "スペック"],
            "action": "get_status",
            "needs_reasoning": False,
        },
        "repos": {
            "keywords": ["github", "repo", "repository", "commit",
                         "リポジトリ", "コミット"],
            "action": "list_repos",
            "needs_reasoning": False,
        },
        "files": {
            "keywords": ["drive", "file", "folder", "gdrive",
                         "ドライブ", "ファイル"],
            "action": "list_files",
            "needs_reasoning": False,
        },
        "notify": {
            "keywords": ["discord", "notify", "send", "alert",
                         "通知", "送信", "連絡"],
            "action": "send_notification",
            "needs_reasoning": False,
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
                        "needs_reasoning": Noul(
                            "Does this require creative writing, complex reasoning, or code generation?"
                        ),
                        "urgency": Score(0, 5, "How urgent? 0=normal, 5=critical"),
                    }
                )
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                return {
                    "source": "Jev API (TypeSafe AI)",
                    "action": res.answers.action.value,
                    "needs_reasoning": res.answers.needs_reasoning.value,
                    "urgency": res.answers.urgency.value,
                    "latency_ms": elapsed_ms,
                }
            except Exception as e:
                print(f"[Jev API fallback: {e}]", file=sys.stderr)

        # Keyword fallback (still 0.01ms)
        prompt_lower = user_input.lower()
        for rule in self.ROUTING_RULES.values():
            if any(kw in prompt_lower for kw in rule["keywords"]):
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                return {
                    "source": "Jev Classifier (keyword fallback)",
                    "action": rule["action"],
                    "needs_reasoning": rule["needs_reasoning"],
                    "urgency": 1,
                    "latency_ms": elapsed_ms,
                }

        # Default: escalate to System 2
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "source": "Jev Classifier (keyword fallback)",
            "action": "llm_reasoning",
            "needs_reasoning": True,
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

    def __init__(self, system_instruction: str = None):
        self.client = genai.Client(api_key=GEMINI_API_KEY) if HAS_GENAI and GEMINI_API_KEY else None
        self.system_instruction = system_instruction or (
            "You are a helpful AI assistant. Respond concisely and accurately."
        )

    def generate(self, user_input: str, conversation_history: list = None) -> str:
        """Generate a response using Gemini with optional conversation context."""
        if not self.client:
            return "Error: GEMINI_API_KEY is not configured."

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
        for model_name in self.MODELS:
            try:
                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                    )
                )
                return resp.text
            except Exception as e:
                print(f"[{model_name} failed: {e}, trying fallback...]", file=sys.stderr)
                time.sleep(1)

        return "LLM temporarily unavailable. Please try again."


class DualProcessRouter:
    """
    Unified orchestrator combining System 1 (Jev) + System 2 (Gemini).
    
    Usage:
        router = DualProcessRouter()
        result = router.process("What's the server status?")
        # → System 1 instant route, no LLM call
        
        result = router.process("Design a REST API for this service")
        # → System 1 routes to System 2, Gemini generates response
    """

    def __init__(self, tool_handlers: dict = None, system_instruction: str = None):
        """
        Args:
            tool_handlers: Dict mapping action names to callable functions.
                           e.g. {"get_status": my_status_func, "list_repos": my_repos_func}
            system_instruction: Custom system prompt for the Gemini reasoner.
        """
        self.system1 = JevClassifier()
        self.system2 = GeminiReasoner(system_instruction=system_instruction)
        self.tool_handlers = tool_handlers or {}

    def process(self, user_input: str, conversation_history: list = None) -> dict:
        """
        Process a user request through the dual-process pipeline.
        
        Returns:
            {
                "input": str,
                "decision": { ... System 1 classification ... },
                "system2_engaged": bool,
                "total_latency_ms": float,
                "output": str,
            }
        """
        total_start = time.perf_counter()

        # --- System 1: Instant classification ---
        decision = self.system1.classify(user_input)
        action = decision["action"]
        system2_engaged = False
        output = ""

        if action != "llm_reasoning" and action in self.tool_handlers:
            # Direct tool execution (no LLM needed)
            try:
                output = self.tool_handlers[action](user_input)
            except Exception as e:
                output = f"Tool execution error: {e}"

        elif action == "llm_reasoning" or action not in self.tool_handlers:
            # --- System 2: Deep reasoning ---
            system2_engaged = True
            output = self.system2.generate(user_input, conversation_history)

        total_ms = round((time.perf_counter() - total_start) * 1000, 2)

        return {
            "input": user_input,
            "decision": decision,
            "system2_engaged": system2_engaged,
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

    router = DualProcessRouter(tool_handlers=handlers)

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
        print(f"   Needs reasoning: {d['needs_reasoning']}")
        print(f"   Latency: {d['latency_ms']} ms")
        print(f"🧠 System 2 engaged: {'YES' if result['system2_engaged'] else 'NO ($0)'}")
        print(f"⏱️  Total: {result['total_latency_ms']} ms")
        print(f"📤 Output: {result['output'][:200]}")


if __name__ == "__main__":
    main()
