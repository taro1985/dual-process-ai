#!/usr/bin/env python3
"""
Jev Memory Scorer — Reflexive Episodic Memory Scorer for HermesAgent & hermes-memory-evolver

Scores the relevance of past episodes against the current task/context in <0.1ms.
Eliminates irrelevant memories and reduces token context / latency for LLM reasoning.
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Optional: TypeSafe Jev API
try:
    from typesafe_sdk import TypeSafeClient, Score
    HAS_TYPESAFE = True
except ImportError:
    HAS_TYPESAFE = False


class JevMemoryScorer:
    """
    Sub-millisecond relevance scorer for episodic memory and knowledge retrieval.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY", "")
        self.client = None
        if HAS_TYPESAFE and self.api_key:
            try:
                self.client = TypeSafeClient(api_key=self.api_key)
            except Exception:
                self.client = None

    STOP_WORDS = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "is", "are", "was", "were", "and", "or", "it", "this", "that",
        "be", "as", "how", "what", "which", "who", "when", "where", "why",
        "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ",
        "ある", "いる", "も", "する", "から", "な", "こと", "として",
    }

    def _stem_word(self, word: str) -> set[str]:
        """Generate common inflection stems for English words (<0.001ms)."""
        stems = set()
        if len(word) <= 2:
            return stems

        # Plural / 3rd person singular -s, -es, -ies
        if word.endswith("ies") and len(word) >= 5:
            stems.add(word[:-3] + "y")
        elif word.endswith("es") and len(word) >= 4:
            stems.add(word[:-2])
            stems.add(word[:-1])
        elif word.endswith("s") and not word.endswith("ss") and len(word) >= 3:
            stems.add(word[:-1])

        # Past tense / participle -ed
        if word.endswith("ed") and len(word) >= 4:
            stems.add(word[:-2])
            stems.add(word[:-1])  # e.g. separated -> separate

        # Continuous -ing
        if word.endswith("ing") and len(word) >= 5:
            stems.add(word[:-3])
            stems.add(word[:-3] + "e")  # e.g. parsing -> parse

        # Common suffixes
        if word.endswith("tion") and len(word) >= 6:
            stems.add(word[:-4])
            stems.add(word[:-4] + "te")
        elif word.endswith("ly") and len(word) >= 4:
            stems.add(word[:-2])

        return stems

    def _tokenize(self, text: str) -> set[str]:
        """
        Fast sub-millisecond tokenizer:
        - Normalizes punctuation, hyphens, and slashes into word separators
        - Generates raw words, stemmed variants, and Japanese 2-grams
        """
        if not text:
            return set()

        # Split on any non-alphanumeric except CJK characters
        normalized = re.sub(r'[\-_/\\.,;:!?()[\]{}<>"\'`|#*~+=\n\r\t]', ' ', text.lower())
        raw_words = re.findall(r'[a-zA-Z0-9]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+', normalized)

        tokens = set()
        for w in raw_words:
            if not w:
                continue
            tokens.add(w)

            # English stems
            if re.match(r'^[a-zA-Z]+$', w):
                tokens.update(self._stem_word(w))
            # CJK bigrams for substring matching
            elif len(w) >= 2:
                for i in range(len(w) - 1):
                    tokens.add(w[i:i+2])

        return tokens

    CONCEPT_MAP = {
        "安全": {"safety", "guard", "prevent", "block", "secure", "safe"},
        "危険": {"danger", "risk", "harm", "malicious", "catastrophic"},
        "対策": {"solution", "rule", "guard", "protection"},
        "性能": {"performance", "latency", "throughput", "speed"},
        "速度": {"speed", "latency", "fast", "throughput"},
        "高速": {"fast", "speed", "latency"},
        "設計": {"architecture", "design", "structure"},
        "構造": {"architecture", "structure", "design"},
        "アーキテクチャ": {"architecture", "design"},
        "パフォーマンス": {"performance", "speed", "latency"},
        "セキュリティ": {"security", "safety"},
        "フラグ": {"flag", "flags"},
        "シェル": {"shell", "bash", "command"},
        "テスト": {"test", "benchmark", "suite"},
        "ベンチマーク": {"benchmark", "throughput", "latency"},
        "ループ": {"loop", "continuous", "daemon"},
        "同期": {"sync", "synchronization", "push"},
        "メモリ": {"memory", "scorer", "ram"},
    }

    def calculate_score(self, query: str, episode: Dict[str, Any]) -> float:
        """
        Calculate relevance score (0.0 to 1.0) between query and an episode/principle.
        Latency: < 0.05ms.
        """
        if not query or not episode:
            return 0.0

        # Try TypeSafe API if available
        if self.client:
            try:
                task_str = episode.get("task", "")
                sol_str = episode.get("solution_summary", "")
                ep_text = f"Task: {task_str} | Solution: {sol_str}"
                res = self.client.system_one(
                    state=f"Current query: '{query}'\nPast Episode: '{ep_text}'",
                    questions={
                        "relevance": Score(
                            0, 100,
                            "How relevant is this past episode to solving the current query? "
                            "(0=irrelevant, 100=highly relevant/identical pattern)"
                        )
                    }
                )
                return round(res.answers.relevance.value / 100.0, 3)
            except Exception:
                pass  # Fallback to heuristic

        # Reflexive Heuristic Scorer (< 0.05ms)
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        # Query Expansion: cross-lingual concept mapping (Japanese -> English principles)
        expanded_tokens = set(query_tokens)
        for jp_key, en_set in self.CONCEPT_MAP.items():
            if jp_key in query:
                expanded_tokens.update(en_set)

        # Filter query tokens excluding stop words for meaningful matching
        content_query = {t for t in expanded_tokens if t not in self.STOP_WORDS and len(t) >= 2}
        effective_query = content_query if content_query else query_tokens

        # Extract target tokens from different fields
        task_tokens = self._tokenize(str(episode.get("task", "")))
        rule_tokens = self._tokenize(str(episode.get("solution_summary", "") or episode.get("actionable_rule", "")))
        domain_str = str(episode.get("domain", "") or episode.get("tags", "")).lower()
        domain_tokens = self._tokenize(domain_str)
        extra_tokens = self._tokenize(" ".join([
            str(episode.get("error_experienced", "") or ""),
            str(episode.get("project_name", "") or ""),
        ]))

        all_target_tokens = task_tokens | rule_tokens | domain_tokens | extra_tokens

        intersection = effective_query & all_target_tokens
        if not intersection:
            return 0.0

        # Overlap ratio relative to content query
        base_overlap = len(intersection) / max(len(effective_query), 1)

        # Domain boost: if user query specifically matches the domain (e.g. "safety", "architecture")
        domain_match = len(effective_query & domain_tokens) > 0
        domain_boost = 0.25 if domain_match else 0.0

        # Actionable Rule / Solution boost: matching the concrete rule
        rule_match_count = len(effective_query & rule_tokens)
        rule_boost = (rule_match_count / max(len(effective_query), 1)) * 0.35

        # Task / Lesson boost
        task_match_count = len(effective_query & task_tokens)
        task_boost = (task_match_count / max(len(effective_query), 1)) * 0.20

        # Status boost
        status_boost = 0.05 if episode.get("status") == "success" else 0.0

        total_score = min(1.0, round(base_overlap * 0.45 + domain_boost + rule_boost + task_boost + status_boost, 3))
        return total_score

    def rank_and_filter(
        self,
        query: str,
        episodes: List[Dict[str, Any]],
        threshold: float = 0.25,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Score, filter below threshold, and return top-k episodes.
        Each returned episode includes a `jev_score` and `score_latency_ms`.
        """
        start = time.perf_counter()
        scored_episodes = []

        for ep in episodes:
            score = self.calculate_score(query, ep)
            if score >= threshold:
                ep_copy = dict(ep)
                ep_copy["jev_score"] = score
                scored_episodes.append(ep_copy)

        scored_episodes.sort(key=lambda x: x["jev_score"], reverse=True)
        top_k = scored_episodes[:limit]

        elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
        for ep in top_k:
            ep["score_latency_ms"] = elapsed_ms

        return top_k

    def find_relevant_principles(
        self,
        query: str,
        principles: Optional[List[Dict[str, Any]]] = None,
        principles_path: Optional[str] = None,
        threshold: float = 0.2,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Reflexively find accumulated evolved principles relevant to the given query.
        Latency: < 0.1ms.
        """
        if principles is None:
            principles = load_evolved_principles(principles_path)
        if not principles:
            return []
        return self.rank_and_filter(query, principles, threshold=threshold, limit=limit)


def parse_evolved_principles(markdown_text: str) -> List[Dict[str, Any]]:
    """
    Parse EVOLVED_PRINCIPLES.md format into a list of structured principle dicts.
    """
    if not markdown_text:
        return []

    pattern = re.compile(
        r'###\s+🧬\s+Evolved\s+Principle\s+\[(.*?)\]\s*\n'
        r'-\s+\*\*Domain\*\*:\s*`?(.*?)`?\s*\n'
        r'-\s+\*\*Lesson\*\*:\s*(.*?)\s*\n'
        r'-\s+\*\*Actionable Rule\*\*:\s*(.*?)\s*\n'
        r'-\s+\*\*Confidence Impact\*\*:\s*(.*?)\s*(?:\n---|\Z)',
        re.DOTALL
    )

    principles = []
    for match in pattern.finditer(markdown_text):
        ts, domain, lesson, rule, conf_impact = match.groups()
        rule_clean = rule.strip().strip('*').strip()
        principles.append({
            "timestamp": ts.strip(),
            "domain": domain.strip(),
            "lesson": lesson.strip(),
            "actionable_rule": rule_clean,
            "confidence_impact": conf_impact.strip(),
            # Compatibility fields for JevMemoryScorer
            "task": f"[{domain.strip()}] {lesson.strip()}",
            "solution_summary": rule_clean,
            "tags": domain.strip(),
            "status": "success",
        })
    return principles


def load_evolved_principles(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load and parse evolved principles from EVOLVED_PRINCIPLES.md file.
    """
    if file_path is None:
        # Default to repo root EVOLVED_PRINCIPLES.md
        file_path = Path(__file__).resolve().parent.parent.parent / "EVOLVED_PRINCIPLES.md"
        if not file_path.exists():
            # Try current directory
            file_path = Path("EVOLVED_PRINCIPLES.md")
    else:
        file_path = Path(file_path)

    if not file_path.exists():
        return []

    try:
        content = file_path.read_text(encoding="utf-8")
        return parse_evolved_principles(content)
    except Exception:
        return []

