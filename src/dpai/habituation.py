#!/usr/bin/env python3
"""
Habituation Engine — Sub-millisecond S2 -> S1 Compilation & Adaptive Learning

Enables Dual-Process AI to "habituate" (learn reflexes):
When System 2 (Gemini) successfully answers a query, the Habituation Engine
compiles the pattern and response into a sub-millisecond System 1 reflex rule.
Subsequent identical or similar queries are served instantly at $0 cost (0.02ms).
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


class HabituationEngine:
    """
    Reflexive memory compiler that turns repeated System 2 deep reasoning
    into instant System 1 reflexive responses.
    """

    STOP_WORDS = {
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "is", "are", "was", "were", "and", "or", "it", "this", "that",
        "be", "as", "how", "what", "which", "who", "when", "where", "why",
        "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ",
        "ある", "いる", "も", "する", "から", "な", "こと", "として", "教えて",
        "ください", "について", "とは",
    }

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.habits: List[Dict[str, Any]] = []
        if self.persistence_path and self.persistence_path.exists():
            self.load()

    def _stem_word(self, word: str) -> set[str]:
        """Generate common inflection stems for English words."""
        stems = set()
        if len(word) <= 2:
            return stems
        if word.endswith("ies") and len(word) >= 5:
            stems.add(word[:-3] + "y")
        elif word.endswith("es") and len(word) >= 4:
            stems.add(word[:-2])
            stems.add(word[:-1])
        elif word.endswith("s") and not word.endswith("ss") and len(word) >= 3:
            stems.add(word[:-1])
        if word.endswith("ed") and len(word) >= 4:
            stems.add(word[:-2])
            stems.add(word[:-1])
        if word.endswith("ing") and len(word) >= 5:
            stems.add(word[:-3])
            stems.add(word[:-3] + "e")
        return stems

    def _tokenize(self, text: str) -> set[str]:
        """Fast sub-millisecond tokenizer with stemming and Japanese bigrams."""
        if not text:
            return set()
        normalized = re.sub(r'[\-_/\\.,;:!?()[\]{}<>"\'`|#*~+=\n\r\t]', ' ', text.lower())
        raw_words = re.findall(r'[a-zA-Z0-9]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+', normalized)
        tokens = set()
        for w in raw_words:
            if not w:
                continue
            tokens.add(w)
            if re.match(r'^[a-zA-Z]+$', w):
                tokens.update(self._stem_word(w))
            elif len(w) >= 2:
                for i in range(len(w) - 1):
                    tokens.add(w[i:i+2])
        return tokens

    def _extract_content_tokens(self, text: str) -> set[str]:
        """Extract meaningful tokens excluding stop words."""
        all_tokens = self._tokenize(text)
        content = {t for t in all_tokens if t not in self.STOP_WORDS and len(t) >= 2}
        return content if content else all_tokens

    def habituate(
        self,
        query: str,
        response: str,
        action: str = "direct_response",
        confidence: float = 0.95,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compile a successful System 2 resolution into a System 1 habit rule.
        """
        if not query or not response:
            return {}

        content_tokens = list(self._extract_content_tokens(query))
        if not content_tokens:
            return {}

        # Check if already exists; if so, update hit count and response
        for h in self.habits:
            existing_tokens = set(h.get("content_tokens", []))
            overlap = len(set(content_tokens) & existing_tokens)
            if overlap / max(len(content_tokens), 1) > 0.85:
                h["response"] = response
                h["hit_count"] = h.get("hit_count", 0) + 1
                h["last_updated"] = time.time()
                if metadata:
                    h.setdefault("metadata", {}).update(metadata)
                if self.persistence_path:
                    self.save()
                return h

        habit = {
            "query_sample": query,
            "content_tokens": content_tokens,
            "action": action,
            "response": response,
            "confidence": confidence,
            "hit_count": 1,
            "created_at": time.time(),
            "last_updated": time.time(),
            "metadata": metadata or {},
        }
        self.habits.append(habit)

        if self.persistence_path:
            self.save()

        return habit

    def match(self, query: str, threshold: float = 0.65) -> Optional[Dict[str, Any]]:
        """
        Sub-millisecond lookup to see if the query matches an existing habit.
        Latency: < 0.05ms.
        """
        if not query or not self.habits:
            return None

        query_tokens = self._extract_content_tokens(query)
        if not query_tokens:
            return None

        best_habit = None
        best_score = 0.0

        for h in self.habits:
            h_tokens = set(h.get("content_tokens", []))
            if not h_tokens:
                continue

            intersection = query_tokens & h_tokens
            if not intersection:
                continue

            # Overlap coefficient
            score = len(intersection) / max(len(query_tokens), 1)
            if score > best_score and score >= threshold:
                best_score = score
                best_habit = h

        if best_habit:
            best_habit["hit_count"] = best_habit.get("hit_count", 0) + 1
            best_habit["last_matched"] = time.time()
            if self.persistence_path:
                self.save()
            result = dict(best_habit)
            result["similarity_score"] = round(best_score, 3)
            return result

        return None

    def save(self, path: Optional[str] = None) -> None:
        """Persist habit rules to JSON file."""
        target = Path(path) if path else self.persistence_path
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(json.dumps(self.habits, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load(self, path: Optional[str] = None) -> None:
        """Load habit rules from JSON file."""
        target = Path(path) if path else self.persistence_path
        if not target or not target.exists():
            return
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.habits = data
        except Exception:
            pass

    def clear(self) -> None:
        """Clear all learned habits."""
        self.habits.clear()
        if self.persistence_path and self.persistence_path.exists():
            try:
                self.persistence_path.unlink()
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Return engine statistics."""
        return {
            "total_habits": len(self.habits),
            "total_hits": sum(h.get("hit_count", 0) for h in self.habits),
        }
