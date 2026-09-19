#!/usr/bin/env python3
"""
DPAI Interactive CLI Demo — Fast & Slow AI in Action

Experience Daniel Kahneman's Dual-Process Theory applied to AI systems.
Run directly without external dependencies:
    uv run python demo_cli.py
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from router import DualProcessRouter
from safety_gate import inspect_command
from tool_pruner import MCPToolPruner

# ANSI Colors for Terminal
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_BG_BLUE = "\033[44m\033[97m"

MOCK_CATALOG = [
    {"name": "docker_list_containers", "description": "List all Docker containers"},
    {"name": "docker_restart_container", "description": "Restart a specific container"},
    {"name": "git_pull_repo", "description": "Pull latest git changes"},
    {"name": "sql_query_database", "description": "Query PostgreSQL database"},
    {"name": "send_slack_message", "description": "Send notification to Slack"},
    {"name": "get_weather_forecast", "description": "Get 5-day weather forecast"},
]


def print_banner():
    banner = f"""
{C_CYAN}{C_BOLD}╔════════════════════════════════════════════════════════════════════════════╗
║                  🧠⚡ DPAI (Dual-Process AI) Interactive Demo                ║
║           System 1: Reflexive Jev (0.01ms / $0)  x  System 2: Gemini        ║
╚════════════════════════════════════════════════════════════════════════════╝{C_RESET}
"""
    print(banner)


def run_sample_query(router: DualProcessRouter, pruner: MCPToolPruner, prompt: str):
    print(f"\n{C_BOLD}💬 Input Prompt:{C_RESET} {C_YELLOW}\"{prompt}\"{C_RESET}")

    # 1. Safety Gate Check (System 1 Reflexive Guard)
    gate_start = time.perf_counter()
    safety_result = inspect_command(prompt)
    gate_ms = round((time.perf_counter() - gate_start) * 1000, 3)

    if safety_result.get("decision") == "deny":
        print(f"{C_RED}{C_BOLD}🛡️ [System 1 Safety Gate] EXECUTION BLOCKED ({gate_ms}ms){C_RESET}")
        print(f"   Reason: {safety_result.get('reason')}")
        print(f"   {C_DIM}Result: Catastrophic command denied before LLM consideration. ($0){C_RESET}")
        return

    # 2. Dual-Process Router (Adaptive Threshold)
    res = router.process(prompt)
    decision = res["decision"]
    action = decision["action"]
    conf = res["confidence"]
    thresh = res["threshold"]
    risk = res["risk_level"]
    s2 = res["system2_engaged"]
    latency = res["total_latency_ms"]

    if not s2:
        # System 1 Fast Path
        print(f"{C_GREEN}{C_BOLD}⚡ [System 1 / Jev Fast Path] DIRECT EXECUTION{C_RESET}")
        print(f"   Action:     {C_CYAN}{action}{C_RESET}")
        print(f"   Risk Level: {C_DIM}{risk}{C_RESET} | Calibrated Threshold: {C_DIM}{thresh:.2f}{C_RESET}")
        print(f"   Confidence: {C_GREEN}{conf:.2f} >= {thresh:.2f}{C_RESET}")
        print(f"   Latency:    {C_GREEN}{latency:.2f} ms{C_RESET}")
        print(f"   API Cost:   {C_GREEN}$0.000000{C_RESET}")
        print(f"   Output:     {C_DIM}{res['output'][:120]}...{C_RESET}")
    else:
        # System 2 Reasoning Path
        print(f"{C_MAGENTA}{C_BOLD}🧠 [System 2 / Gemini Reasoning] ESCALATED{C_RESET}")
        print(f"   Reason for Escalation:")
        if "ambiguity" in risk:
            print(f"   • {C_YELLOW}Ambiguity penalty detected (+0.10 threshold: {thresh:.2f}){C_RESET}")
        elif action == "llm_reasoning":
            print(f"   • Complex reasoning required (Confidence: {conf:.2f} < {thresh:.2f})")
        else:
            print(f"   • Confidence ({conf:.2f}) below risk threshold ({thresh:.2f})")

        # Tool Pruning Demonstration
        pruned = pruner.prune(prompt, MOCK_CATALOG, top_k=2)
        pruned_names = [t["name"] for t in pruned]
        print(f"   • {C_CYAN}MCP Dynamic Pruning:{C_RESET} Retained {len(pruned)}/6 tools {pruned_names}")
        print(f"   Latency:    {C_MAGENTA}{latency:.2f} ms{C_RESET}")
        print(f"   Cost:       Standard Gemini token rate")
        print(f"   Output:     {C_DIM}{res['output'][:140]}...{C_RESET}")


def interactive_mode():
    print_banner()

    # Handlers for demo
    handlers = {
        "get_status": lambda inp: '{"host": "vaio-server", "cpu": "12%", "memory": "48%", "status": "healthy"}',
        "docker_status": lambda inp: '{"containers": [{"name": "vaio-mcp", "status": "Up 14 hours"}]}',
        "docker_restart": lambda inp: '{"status": "success", "container": "vaio-mcp", "message": "Restarted"}',
        "git_pull": lambda inp: '{"status": "success", "output": "Already up to date."}',
    }
    router = DualProcessRouter(tool_handlers=handlers, use_adaptive_threshold=True)
    pruner = MCPToolPruner()

    presets = [
        ("1", "サーバーのステータス見せて", "定型参照タスク (System 1 / 低リスク / 0.01ms)"),
        ("2", "docker restart vaio-mcp", "定型変更タスク (System 1 / 中リスク / 高確信度)"),
        ("3", "docker restart vaio-mcp かも？", "曖昧指示 (曖昧性ペナルティ加算 -> System 2エスカレーション)"),
        ("4", "rm -rf /", "危険コマンド (System 1 セーフティゲート即時物理遮断)"),
        ("5", "マイクロサービスアーキテクチャの設計案を考えて", "高度推論タスク (System 2 / Gemini 熟慮)"),
        ("6", "データベースのクエリを最適化したい", "MCPツール動的枝刈り (MCPToolPruner)"),
    ]

    print(f"{C_BOLD}Select a preset scenario or type your own prompt:{C_RESET}\n")
    for key, prompt, desc in presets:
        print(f"  {C_CYAN}[{key}]{C_RESET} {C_BOLD}{prompt}{C_RESET}")
        print(f"      {C_DIM}→ {desc}{C_RESET}")
    print(f"  {C_CYAN}[0]{C_RESET} 自由入力モード (Type arbitrary prompt)")
    print(f"  {C_CYAN}[q]{C_RESET} 終了 (Quit)\n")

    while True:
        try:
            choice = input(f"{C_BOLD}DPAI ❯ {C_RESET}").strip()
            if not choice:
                continue
            if choice.lower() in ["q", "quit", "exit"]:
                print(f"\n{C_CYAN}Exiting DPAI Demo. Fast brain decides, deep brain thinks!{C_RESET}")
                break

            matched_preset = next((p for k, p, _ in presets if k == choice), None)
            if matched_preset:
                run_sample_query(router, pruner, matched_preset)
                print("\n" + "─" * 70)
            elif choice == "0":
                custom = input(f"{C_YELLOW}Enter prompt: {C_RESET}").strip()
                if custom:
                    run_sample_query(router, pruner, custom)
                    print("\n" + "─" * 70)
            else:
                # Direct prompt
                run_sample_query(router, pruner, choice)
                print("\n" + "─" * 70)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    interactive_mode()
