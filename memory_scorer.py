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

    def _tokenize(self, text: str) -> set[str]:
        """Simple, fast tokenizer for Japanese and English words."""
        if not text:
            return set()
        # Extract alphanumeric words and Japanese token chunks
        words = re.findall(r'[a-zA-Z0-9_\-]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+', text.lower())
        return set(words)

    def calculate_score(self, query: str, episode: Dict[str, Any]) -> float:
        """
        Calculate relevance score (0.0 to 1.0) between query and an episode.
        Latency: < 0.1ms (local fallback) or fast RLCD API.
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

        target_text = " ".join([
            str(episode.get("task", "")),
            str(episode.get("solution_summary", "")),
            str(episode.get("error_experienced", "") or ""),
            str(episode.get("tags", "") or ""),
            str(episode.get("project_name", "") or ""),
        ])
        target_tokens = self._tokenize(target_text)
        if not target_tokens:
            return 0.0

        # Jaccard / Overlap coefficient with priority weighting for task match
        intersection = query_tokens & target_tokens
        if not intersection:
            return 0.0

        task_tokens = self._tokenize(str(episode.get("task", "")))
        task_match_count = len(query_tokens & task_tokens)

        # Base overlap ratio
        overlap_score = len(intersection) / len(query_tokens)

        # Boost if task itself matches directly
        task_boost = (task_match_count / len(query_tokens)) * 0.3

        # Status boost: successful episodes get slight preference
        status_boost = 0.05 if episode.get("status") == "success" else 0.0

        total_score = min(1.0, round(overlap_score * 0.7 + task_boost + status_boost, 3))
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
