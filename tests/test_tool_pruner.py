#!/usr/bin/env python3
"""
Tests for Dynamic MCP Tool Pruning (System 1 Tool Selector).
"""

import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tool_pruner import MCPToolPruner
from router import DualProcessRouter


MOCK_MCP_TOOLS = [
    {
        "name": "docker_list_containers",
        "description": "List all running and exited Docker containers",
        "inputSchema": {"type": "object", "properties": {"all": {"type": "boolean"}}},
    },
    {
        "name": "docker_restart_container",
        "description": "Restart a specific docker container by ID or name",
        "inputSchema": {"type": "object", "properties": {"container_id": {"type": "string"}}},
    },
    {
        "name": "git_pull_repo",
        "description": "Fetch and pull latest changes from remote Git repository",
        "inputSchema": {"type": "object", "properties": {"branch": {"type": "string"}}},
    },
    {
        "name": "git_commit_changes",
        "description": "Record changes to the repository with a commit message",
        "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}},
    },
    {
        "name": "read_filesystem_file",
        "description": "Read contents of a file on local filesystem",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    {
        "name": "sql_query_database",
        "description": "Execute a SELECT query against the PostgreSQL database",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
    {
        "name": "send_slack_message",
        "description": "Send a chat notification to a Slack channel",
        "inputSchema": {"type": "object", "properties": {"channel": {"type": "string"}, "text": {"type": "string"}}},
    },
    {
        "name": "get_weather_forecast",
        "description": "Fetch current weather and 5-day forecast for a city",
        "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
]


def test_mcp_tool_pruner_relevance():
    pruner = MCPToolPruner()

    # Query for Docker operations
    docker_tool = MOCK_MCP_TOOLS[0]  # docker_list_containers
    weather_tool = MOCK_MCP_TOOLS[7]  # get_weather_forecast

    score_docker = pruner.calculate_tool_relevance("Show me all docker containers", docker_tool)
    score_weather = pruner.calculate_tool_relevance("Show me all docker containers", weather_tool)

    assert score_docker > 0.4, f"Docker tool score should be high, got {score_docker}"
    assert score_weather < 0.1, f"Weather tool score should be near zero, got {score_weather}"


def test_mcp_tool_pruner_prune_top_k():
    pruner = MCPToolPruner()

    # From 8 diverse tools, prune down to top 2 for git task
    prompt = "Pull the latest commits from the git main branch"
    pruned = pruner.prune(prompt, MOCK_MCP_TOOLS, top_k=2)

    assert len(pruned) <= 2
    tool_names = [t["name"] for t in pruned]
    assert "git_pull_repo" in tool_names
    assert "_prune_score" in pruned[0]
    assert "_prune_latency_ms" in pruned[0]
    assert pruned[0]["_prune_latency_ms"] < 5.0  # sub-millisecond execution


def test_dual_process_router_integrates_tool_pruning():
    # Setup router with MCP tools catalog and System 2 escalation task
    router = DualProcessRouter(mcp_tools=MOCK_MCP_TOOLS, threshold=0.85)

    # Complex request that requires System 2 reasoning
    complex_prompt = "Compare database schema performance and run sql queries to optimize indexes"
    result = router.process(complex_prompt)

    # Escalated to System 2
    assert result["system2_engaged"]
    # Pruned tools should contain sql_query_database
    pruned = result["pruned_tools"]
    assert "sql_query_database" in pruned
    # Unrelated tools should be pruned away
    assert "get_weather_forecast" not in pruned
    assert "send_slack_message" not in pruned
