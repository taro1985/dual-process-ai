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
from typing import Optional, Any, Dict, List, Tuple
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
    from .tool_pruner import MCPToolPruner
    HAS_PRUNER = True
except ImportError:
    try:
        from tool_pruner import MCPToolPruner
        HAS_PRUNER = True
    except ImportError:
        HAS_PRUNER = False


# Continual Learning / Episodic Principles Feedback (System 3 -> System 1)
try:
    from .memory_scorer import JevMemoryScorer, load_evolved_principles
    HAS_MEMORY_SCORER = True
except ImportError:
    try:
        from memory_scorer import JevMemoryScorer, load_evolved_principles
        HAS_MEMORY_SCORER = True
    except ImportError:
        HAS_MEMORY_SCORER = False

# Habituation Engine (System 2 -> System 1 Compilation)
try:
    from .habituation import HabituationEngine
    HAS_HABITUATION = True
except ImportError:
    try:
        from habituation import HabituationEngine
        HAS_HABITUATION = True
    except ImportError:
        HAS_HABITUATION = False

# Type-Safe Pydantic Data Models (Phase 3)
try:
    from .models import RoutingDecision, ProcessResult
    HAS_MODELS = True
except ImportError:
    try:
        from models import RoutingDecision, ProcessResult
        HAS_MODELS = True
    except ImportError:
        HAS_MODELS = False

# Binary-Native Protocol & Bitwise Latent Matcher
try:
    from .binary_protocol import (
        BinaryActionPacket,
        BitwiseLatentMatcher,
        ACTION_GET_STATUS,
        ACTION_DOCKER_STATUS,
        ACTION_DOCKER_RESTART,
        ACTION_GIT_PULL,
        ACTION_LIST_REPOS,
        ACTION_LIST_FILES,
        ACTION_NOTIFY,
        ACTION_HABIT_REFLEX,
        ACTION_SAFETY_BLOCKED,
        ACTION_ESCALATE_S2,
        ACTION_TO_NAME,
        NAME_TO_ACTION,
        FLAG_SAFETY_BLOCKED,
        FLAG_HABIT_HIT,
        FLAG_AMBIGUITY_PENALTY,
        FLAG_FALLBACK_ACTIVE,
    )
    HAS_BINARY_PROTOCOL = True
except ImportError:
    try:
        from binary_protocol import (
            BinaryActionPacket,
            BitwiseLatentMatcher,
            ACTION_GET_STATUS,
            ACTION_DOCKER_STATUS,
            ACTION_DOCKER_RESTART,
            ACTION_GIT_PULL,
            ACTION_LIST_REPOS,
            ACTION_LIST_FILES,
            ACTION_NOTIFY,
            ACTION_HABIT_REFLEX,
            ACTION_SAFETY_BLOCKED,
            ACTION_ESCALATE_S2,
            ACTION_TO_NAME,
            NAME_TO_ACTION,
            FLAG_SAFETY_BLOCKED,
            FLAG_HABIT_HIT,
            FLAG_AMBIGUITY_PENALTY,
            FLAG_FALLBACK_ACTIVE,
        )
        HAS_BINARY_PROTOCOL = True
    except ImportError:
        HAS_BINARY_PROTOCOL = False





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

    def __init__(self, habit_engine: Optional[Any] = None):
        self.client = None
        self.habit_engine = habit_engine
        if HAS_TYPESAFE and TYPESAFE_API_KEY:
            self.client = TypeSafeClient(api_key=TYPESAFE_API_KEY)

    def classify(self, user_input: str) -> dict:
        """Classify user intent in sub-millisecond time."""
        start = time.perf_counter()

        # Helper to return RoutingDecision or dict
        def make_decision(data: dict) -> Any:
            if HAS_MODELS:
                return RoutingDecision(**data)
            return data

        # Check Habituation Engine first (Learned S1 Reflexes, < 0.05ms)
        if self.habit_engine:
            matched_habit = self.habit_engine.match(user_input, threshold=0.65)
            if matched_habit:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                return make_decision({
                    "source": "Habituation Engine (Learned S1 Reflex)",
                    "action": matched_habit.get("action", "habituated_response"),
                    "confidence": matched_habit.get("confidence", 0.96),
                    "urgency": 1,
                    "habit_response": matched_habit.get("response", ""),
                    "is_habituated": True,
                    "latency_ms": elapsed_ms,
                })

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
                return make_decision({
                    "source": "Jev API (TypeSafe AI)",
                    "action": res.answers.action.value,
                    "confidence": conf_score,
                    "urgency": res.answers.urgency.value,
                    "latency_ms": elapsed_ms,
                })
            except Exception as e:
                print(f"[Jev API fallback: {e}]", file=sys.stderr)

        # Keyword fallback (still 0.01ms, degraded mode)
        prompt_lower = user_input.lower()
        for rule in self.ROUTING_RULES.values():
            if any(kw in prompt_lower for kw in rule["keywords"]):
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                return make_decision({
                    "source": "Jev Classifier (keyword fallback)",
                    "action": rule["action"],
                    "confidence": 0.95,
                    "urgency": 1,
                    "latency_ms": elapsed_ms,
                })

        # Default: escalate to System 2
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return make_decision({
            "source": "Jev Classifier (keyword fallback)",
            "action": "llm_reasoning",
            "confidence": 0.20,
            "urgency": 1,
            "latency_ms": elapsed_ms,
        })


class System1_5FallbackReasoner:
    """
    System 1.5: Offline & Quota-Resistant Rule-Based Reasoner.
    
    Triggered when System 2 encounters rate limits (429 RESOURCE_EXHAUSTED)
    or network errors. Formulates structured guidance based on recalled principles
    and intent heuristics without relying on external cloud APIs.
    """

    @staticmethod
    def fallback_reason(user_input: str, principles: list = None, error_context: str = "") -> str:
        prompt_lower = user_input.lower()
        parts = ["⚠️ [System 1.5 Fallback Active: Gemini Quota/Offline Mode]"]

        # 1. Concrete intent-based heuristic recommendation
        action_guide = None
        if any(w in prompt_lower for w in ["port", "ポート", "network", "ネット"]):
            action_guide = "推奨アクション（ポート・ネットワーク確認）:\n- リッスン中ポート: `sudo ss -tulpn`\n- 接続状況: `ss -s`"
        elif any(w in prompt_lower for w in ["git", "push", "remote", "リポジトリ"]):
            action_guide = "推奨アクション（Gitリポジトリ操作）:\n- 状態確認: `git status`\n- リモート同期: `git push private <branch>`"
        elif any(w in prompt_lower for w in ["docker", "container", "コンテナ"]):
            action_guide = "推奨アクション（Docker操作）:\n- コンテナ一覧: `docker ps`\n- ログ確認: `docker logs <container_id>`"
        elif any(w in prompt_lower for w in ["cpu", "memory", "メモリ", "負荷", "load"]):
            action_guide = "推奨アクション（システム負荷確認）:\n- メモリ: `free -h`\n- CPU/プロセス: `top -b -n 1 | head -15`"

        if action_guide:
            parts.append(action_guide)

        # 2. Integrate recalled principles as safety / architecture guardrails
        if principles:
            rules = [f"- {p}" for p in principles if p]
            if rules:
                parts.append("関連する蓄積原則（Evolved Principles）に基づくガイダンス:")
                parts.extend(rules)

        if len(parts) == 1:
            # Default fallback when no specific heuristic or principle matched
            parts.append(
                f"リクエスト '{user_input}' は深層推論（System 2）が必要ですが、現在外部LLM APIが一時利用制限中です。\n"
                f"ローカルSystem 1反射およびキャッシュは正常稼働しています。"
            )

        return "\n".join(parts)


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

        # System 1: Dynamically prune tools first (<0.1ms, independent of LLM API key)
        genai_tools = None
        if tools and self.pruner and self.enable_tool_pruning:
            self.last_pruned_tools = self.pruner.prune(user_input, tools, top_k=3, min_score=0.15)
            if self.last_pruned_tools:
                genai_tools = self.pruner.to_genai_function_declarations(self.last_pruned_tools)
        elif tools:
            self.last_pruned_tools = tools

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
        principles_path: str = None,
        load_principles: bool = True,
        enable_habituation: bool = True,
        habit_persistence_path: str = None,
        habit_engine: Any = None,
    ):
        """
        Args:
            tool_handlers: Dict mapping action names to callable functions.
            system_instruction: Custom system prompt for the Gemini reasoner.
            threshold: Default calibrated confidence threshold (0.0 - 1.0).
            use_adaptive_threshold: If True, dynamically adjusts threshold based on action risk & input ambiguity.
            mcp_tools: Optional catalog of available MCP tool definitions for dynamic pruning.
            principles_path: Path to EVOLVED_PRINCIPLES.md (System 3 memory).
            load_principles: If True, loads and utilizes accumulated evolved principles.
            enable_habituation: If True, compiles successful System 2 reasoning into System 1 reflexes.
            habit_persistence_path: Path to persist habituation rules JSON.
            habit_engine: Optional existing HabituationEngine instance.
        """
        # S2 -> S1 Habituation Engine
        self.enable_habituation = enable_habituation
        if habit_engine is not None:
            self.habit_engine = habit_engine
        elif enable_habituation and HAS_HABITUATION:
            self.habit_engine = HabituationEngine(persistence_path=habit_persistence_path)
        else:
            self.habit_engine = None

        self.system1 = JevClassifier(habit_engine=self.habit_engine)
        self.classifier = self.system1  # Convenient alias
        self.system2 = GeminiReasoner(system_instruction=system_instruction)
        self.tool_handlers = tool_handlers or {}
        self.threshold = threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.mcp_tools = mcp_tools or []

        # System 3 -> System 1/2 Closed Feedback Loop
        self.memory_scorer = JevMemoryScorer() if HAS_MEMORY_SCORER else None
        self.principles = []
        if load_principles and HAS_MEMORY_SCORER:
            self.principles = load_evolved_principles(principles_path)

        # Binary-Native Bitwise Latent Matcher
        self.binary_matcher = BitwiseLatentMatcher() if HAS_BINARY_PROTOCOL else None

    def classify(self, user_input: str) -> dict:
        """Convenient shortcut to classify input directly via System 1."""
        return self.system1.classify(user_input)

    def classify_packet(self, user_input: str) -> Any:
        """
        Classify input directly into a 16-byte fixed-size binary packet.
        Latency: < 0.01ms (Zero string-formatting overhead).
        """
        if not HAS_BINARY_PROTOCOL:
            raise RuntimeError("Binary protocol is not available.")

        # 1. Fast Bitwise Latent Matching for Habits
        flags = 0
        latent_hash = BitwiseLatentMatcher.compute_simhash(user_input)
        if self.binary_matcher:
            habit_hit = self.binary_matcher.match_fast(user_input)
            if habit_hit:
                return BinaryActionPacket(
                    action_code=ACTION_HABIT_REFLEX,
                    confidence=habit_hit.get("similarity_score", 0.96),
                    flags=FLAG_HABIT_HIT,
                    latent_hash=latent_hash,
                )

        # 2. System 1 Reflex
        decision = self.system1.classify(user_input)
        action_name = decision.get("action", "llm_reasoning")
        action_code = NAME_TO_ACTION.get(action_name, ACTION_ESCALATE_S2)
        confidence = decision.get("confidence", 0.0)

        # 3. Check Ambiguity Patterns
        for pattern in self.AMBIGUITY_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                flags |= FLAG_AMBIGUITY_PENALTY
                break

        return BinaryActionPacket(
            action_code=action_code,
            confidence=confidence,
            flags=flags,
            latent_hash=latent_hash,
        )

    def process_binary(self, packet_bytes: bytes) -> bytes:
        """
        Process a 16-byte raw binary packet and return a 16-byte response packet.
        Eliminates human text/JSON IPC overhead entirely.
        """
        if not HAS_BINARY_PROTOCOL:
            raise RuntimeError("Binary protocol is not available.")

        in_packet = BinaryActionPacket.unpack(packet_bytes)
        action_code = in_packet.action_code
        confidence = in_packet.confidence
        flags = in_packet.flags

        # If already classified as habit or safe fast path
        effective_threshold = self.threshold
        if flags & FLAG_AMBIGUITY_PENALTY:
            effective_threshold = min(0.99, effective_threshold + 0.10)

        out_action_code = action_code
        out_flags = flags

        if confidence >= effective_threshold and action_code != ACTION_ESCALATE_S2:
            # S1 Reflex execution permitted
            out_flags &= ~FLAG_FALLBACK_ACTIVE
        else:
            # Escalate to System 2
            out_action_code = ACTION_ESCALATE_S2

        out_packet = BinaryActionPacket(
            action_code=out_action_code,
            confidence=confidence,
            flags=out_flags,
            latent_hash=in_packet.latent_hash,
        )
        return out_packet.pack()

    def habituate(self, query: str, response: str, confidence: float = 0.96) -> dict:
        """Manually or explicitly habituate a query-response pair into System 1."""
        res = {}
        if self.habit_engine:
            res = self.habit_engine.habituate(query, response, confidence=confidence)
        if self.binary_matcher:
            self.binary_matcher.register_habit(query, response)
        return res

    def get_habits(self) -> list:
        """Get all learned habit rules."""
        return self.habit_engine.habits if self.habit_engine else []

    def clear_habits(self) -> None:
        """Clear all learned habits."""
        if self.habit_engine:
            self.habit_engine.clear()
        if self.binary_matcher:
            self.binary_matcher.fingerprints.clear()

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
        Process a user request through the dual-process pipeline with adaptive calibration,
        closed-loop continual learning from evolved principles, and S2 -> S1 habituation.
        """
        total_start = time.perf_counter()

        # --- System 1: Instant classification & calibrated confidence ---
        decision = self.system1.classify(user_input)
        action = decision["action"]
        confidence = decision.get("confidence", 0.0)
        system2_engaged = False
        is_fallback = False
        output = ""

        # Check if already habituated (Learned S1 Reflex: < 0.05ms)
        if decision.get("is_habituated"):
            output = decision.get("habit_response", "")
            total_ms = round((time.perf_counter() - total_start) * 1000, 2)
            res_data = {
                "input": user_input,
                "decision": decision,
                "confidence": confidence,
                "threshold": self.threshold,
                "risk_level": "habituated_reflex",
                "system2_engaged": False,
                "is_habituated": True,
                "is_fallback": False,
                "pruned_tools": [],
                "principles": [],
                "total_latency_ms": total_ms,
                "output": output,
            }
            if HAS_MODELS:
                return ProcessResult(**res_data)
            return res_data

        # System 3 -> System 1: Check for relevant evolved principles (< 0.05ms)
        relevant_principles = []
        if self.memory_scorer and self.principles:
            relevant_principles = self.memory_scorer.find_relevant_principles(
                user_input, principles=self.principles, threshold=0.15, limit=2
            )

        # Determine effective threshold (Adaptive vs Static)
        if self.use_adaptive_threshold:
            effective_threshold, risk_level = self.get_adaptive_threshold(action, user_input)
            # If safety principle matches, raise threshold defensively
            for p in relevant_principles:
                if p.get("domain") == "safety" or "raises" in p.get("confidence_impact", "").lower():
                    effective_threshold = min(0.99, round(effective_threshold + 0.05, 2))
                    risk_level += " (evolved safety guard)"
                    break
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
            # --- System 2: Escalate to deep reasoning (with dynamic tool pruning & evolved principles context) ---
            system2_engaged = True
            augmented_history = list(conversation_history) if conversation_history else []
            if relevant_principles:
                rules_text = "\n".join([f"- [{p.get('domain')}]: {p.get('actionable_rule')}" for p in relevant_principles])
                guidance = f"[Accumulated Wisdom / Evolved Principles]:\n{rules_text}\nApply these rules strictly."
                augmented_history.append({"role": "user", "parts": [guidance]})
                augmented_history.append({"role": "model", "parts": ["Understood. I will strictly apply these evolved principles."]})

            output = self.system2.generate(user_input, augmented_history, tools=self.mcp_tools)

            # System 1.5 Fallback: If LLM is unavailable (429 / quota / offline), use local rule-based reasoner
            is_fallback = False
            if not output or output.startswith("LLM temporarily unavailable") or output.startswith("Error:"):
                is_fallback = True
                output = System1_5FallbackReasoner.fallback_reason(
                    user_input,
                    principles=[p.get("actionable_rule") for p in relevant_principles],
                    error_context=output,
                )

            # Auto-Habituation: Compile successful resolution into System 1 habit rule
            if not is_fallback and self.enable_habituation and self.habit_engine:
                if output and len(output) > 5:
                    self.habit_engine.habituate(
                        query=user_input,
                        response=output,
                        action="habituated_response",
                        confidence=0.96,
                    )

        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        pruned_tool_names = [t.get("name") for t in getattr(self.system2, "last_pruned_tools", []) if isinstance(t, dict)]
        principle_rules = [p.get("actionable_rule") for p in relevant_principles]

        res_data = {
            "input": user_input,
            "decision": decision,
            "confidence": confidence,
            "threshold": effective_threshold,
            "risk_level": risk_level,
            "system2_engaged": system2_engaged,
            "is_habituated": False,
            "is_fallback": is_fallback,
            "pruned_tools": pruned_tool_names,
            "principles": principle_rules,
            "total_latency_ms": total_ms,
            "output": output,
        }

        if HAS_MODELS:
            return ProcessResult(**res_data)
        return res_data


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
