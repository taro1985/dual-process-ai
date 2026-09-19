#!/usr/bin/env python3
"""
Tests for JevMemoryScorer and SQLiteMemory Integration.
"""

import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
bot_dir = Path("/home/taro/bot")
for p in [str(root_dir), str(bot_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from memory_scorer import JevMemoryScorer
from memory.sqlite_memory import HermesMemoryDB, Episode


def test_jev_memory_scorer_relevant_episode():
    scorer = JevMemoryScorer()
    query = "Fix Vite build failure missing dependency"
    relevant_ep = {
        "task": "Fix Vite build error with missing autoprefixer",
        "solution_summary": "Installed autoprefixer via npm and updated vite.config.js",
        "status": "success",
        "tags": "vite,build,npm",
    }
    irrelevant_ep = {
        "task": "Setup PostgreSQL database migration",
        "solution_summary": "Created alembic migrations and executed upgrade head",
        "status": "success",
        "tags": "db,postgres,alembic",
    }

    rel_score = scorer.calculate_score(query, relevant_ep)
    irrel_score = scorer.calculate_score(query, irrelevant_ep)

    assert rel_score > 0.4, f"Relevant episode score ({rel_score}) should be > 0.4"
    assert irrel_score < 0.2, f"Irrelevant episode score ({irrel_score}) should be < 0.2"
    assert rel_score > irrel_score


def test_jev_memory_scorer_rank_and_filter():
    scorer = JevMemoryScorer()
    query = "Docker container restart fail"
    episodes = [
        {
            "id": 1,
            "task": "Fix Docker container restart crash loop",
            "solution_summary": "Updated memory limits in docker-compose.yml",
            "status": "success",
            "tags": "docker,container",
        },
        {
            "id": 2,
            "task": "Deploy Next.js app to Vercel",
            "solution_summary": "Configured environment variables on Vercel dashboard",
            "status": "success",
            "tags": "nextjs,vercel",
        },
        {
            "id": 3,
            "task": "Restart Docker daemon and prune dead containers",
            "solution_summary": "systemctl restart docker && docker system prune -f",
            "status": "success",
            "tags": "docker,admin",
        },
    ]

    ranked = scorer.rank_and_filter(query, episodes, threshold=0.2, limit=2)
    assert len(ranked) == 2
    # Both top items should be docker-related
    assert "docker" in ranked[0]["task"].lower()
    assert "docker" in ranked[1]["task"].lower()
    assert ranked[0]["jev_score"] >= ranked[1]["jev_score"]
    assert "score_latency_ms" in ranked[0]
    assert ranked[0]["score_latency_ms"] < 5.0  # Fast sub-millisecond execution


def test_sqlite_memory_integration(tmp_path):
    db_file = tmp_path / "test_hermes_memory.db"
    mem_db = HermesMemoryDB(db_file)

    # Record test episodes
    mem_db.record_episode(Episode(
        id=None,
        timestamp=time.time(),
        workspace_path="/home/taro/dual-process-ai",
        project_name="dual-process-ai",
        task="Optimize router latency with Jev System 1",
        status="success",
        actions_summary="Added JevClassifier routing rules",
        error_experienced=None,
        reflection_notes="Sub-millisecond routing verified",
        solution_summary="Router executes routine actions in 0.01ms",
        tags="router,jev,latency",
    ))
    mem_db.record_episode(Episode(
        id=None,
        timestamp=time.time(),
        workspace_path="/home/taro/unrelated-repo",
        project_name="unrelated-repo",
        task="Create graphic design banner",
        status="success",
        actions_summary="Generated SVG banner",
        error_experienced=None,
        reflection_notes=None,
        solution_summary="Saved banner.svg",
        tags="design,svg",
    ))

    # Search with Jev scoring
    results = mem_db.search_episodes("router latency Jev", limit=1)
    assert len(results) == 1
    assert "router latency" in results[0]["task"].lower()
    assert "jev_score" in results[0]
    assert results[0]["jev_score"] > 0.3
