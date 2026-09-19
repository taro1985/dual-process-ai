#!/usr/bin/env python3
"""
Dual-Process AI Discord Bot

A smartphone-optimized Discord bot powered by the Dual-Process AI pattern:
- System 1 (Jev) routes routine requests instantly ($0)
- System 2 (Gemini 3.8 Flash) handles deep reasoning
- Rich embed cards for structured data
- Short-term conversation memory per channel
"""

import os
import sys
import json
import time
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import re
from dotenv import load_dotenv

import discord
from discord.ext import commands

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

try:
    from dpai import DualProcessRouter, HermesSafetyGate
except ImportError:
    from router import DualProcessRouter
    from hermes_gate import HermesSafetyGate


# =============================================================================
# Configuration
# =============================================================================

# Short-term memory: last N turns per channel
MEMORY_MAX_TURNS = 6

# =============================================================================
# Bot Setup
# =============================================================================

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Channel-level conversation memory
conversation_memories: dict[int, deque] = defaultdict(lambda: deque(maxlen=MEMORY_MAX_TURNS))


# =============================================================================
# Tool Handlers (System 1 Reflexive Remote Operations)
# =============================================================================

def get_status(_input: str) -> str:
    """Check server host, CPU, and memory metrics."""
    import platform
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        return json.dumps({
            "host": platform.node(),
            "cpu_percent": cpu,
            "memory_percent": mem.percent,
            "memory_used_gb": round(mem.used / (1024**3), 2),
            "memory_total_gb": round(mem.total / (1024**3), 2),
        })
    except ImportError:
        return json.dumps({"host": platform.node(), "note": "Install psutil for real metrics"})


def get_docker_status(_input: str) -> str:
    """List docker containers safely."""
    cmd = "docker ps -a --format '{\"id\":\"{{.ID}}\",\"names\":\"{{.Names}}\",\"status\":\"{{.Status}}\",\"image\":\"{{.Image}}\"}'"
    allowed, reason = HermesSafetyGate.inspect(cmd)
    if not allowed:
        return json.dumps({"error": reason})
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            lines = [json.loads(line) for line in res.stdout.strip().splitlines() if line]
            return json.dumps({"status": "success", "containers": lines})
        else:
            return json.dumps({"status": "error", "error": res.stderr.strip() or "Docker command failed"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def restart_docker_container(user_input: str) -> str:
    """Safely restart a specified container."""
    # Check for shell metacharacters / injection attempts anywhere in the input
    if re.search(r'[;&|`$><\\]', user_input):
        return json.dumps({"status": "error", "error": "⚡ [Jev Guard] 不正な文字が含まれています (コマンドインジェクション防止)"})

    match = re.search(r'(?:restart|再起動|リスタート)\s+([a-zA-Z0-9_\-]+)', user_input, re.IGNORECASE)
    if not match:
        tokens = [t for t in re.findall(r'[a-zA-Z0-9_\-]+', user_input)
                  if t.lower() not in ["docker", "restart", "container", "コンテナ", "再起動", "リスタート"]]
        target = tokens[0] if tokens else None
    else:
        target = match.group(1)

    if not target:
        return json.dumps({"status": "error", "error": "対象のコンテナ名を指定してください (例: docker restart vaio-mcp)"})

    # Validate container name against strict regex to prevent command injection
    if not re.fullmatch(r'[a-zA-Z0-9_\-]+', target):
        return json.dumps({"status": "error", "error": "⚡ [Jev Guard] 不正なコンテナ名です (コマンドインジェクション防止)"})

    cmd = f"docker restart {target}"
    allowed, reason = HermesSafetyGate.inspect(cmd)
    if not allowed:
        return json.dumps({"status": "error", "error": reason})

    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            return json.dumps({"status": "success", "container": target, "message": f"Container `{target}` restarted successfully."})
        else:
            return json.dumps({"status": "error", "error": res.stderr.strip() or f"Failed to restart {target}"})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def run_git_pull(_input: str) -> str:
    """Safely pull latest changes from git repository."""
    cmd = "git pull origin main"
    allowed, reason = HermesSafetyGate.inspect(cmd)
    if not allowed:
        return json.dumps({"status": "error", "error": reason})

    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        out = res.stdout.strip() if res.returncode == 0 else res.stderr.strip()
        return json.dumps({"status": "success" if res.returncode == 0 else "error", "output": out})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# Register tool handlers
TOOL_HANDLERS = {
    "get_status": get_status,
    "docker_status": get_docker_status,
    "docker_restart": restart_docker_container,
    "git_pull": run_git_pull,
}

# Initialize the dual-process router
router = DualProcessRouter(tool_handlers=TOOL_HANDLERS)


# =============================================================================
# Rich Embed Cards (smartphone-optimized)
# =============================================================================

def create_status_embed(data: dict, decision_ms: float) -> discord.Embed:
    """Create a color-coded server status card."""
    mem_pct = data.get("memory_percent", 0)

    # Color: green → yellow → red based on memory usage
    if mem_pct < 60:
        color = 0x22C55E  # green
    elif mem_pct < 80:
        color = 0xF59E0B  # yellow
    else:
        color = 0xEF4444  # red

    embed = discord.Embed(
        title="🖥️ Server Status",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(name="Host", value=f"`{data.get('host', 'N/A')}`", inline=True)
    embed.add_field(name="CPU", value=f"{data.get('cpu_percent', 'N/A')}%", inline=True)
    embed.add_field(
        name="Memory",
        value=f"{mem_pct}% ({data.get('memory_used_gb', '?')}/{data.get('memory_total_gb', '?')} GB)",
        inline=True,
    )
    embed.set_footer(text=f"⚡ System 1 (Jev) | {decision_ms}ms | Cost: $0")
    return embed


def create_docker_embed(data: dict, decision_ms: float) -> discord.Embed:
    """Create a smartphone-optimized Docker container status card."""
    if data.get("status") == "error" or "error" in data:
        err_msg = data.get("error", "Unknown error")
        embed = discord.Embed(
            title="🐳 Docker Status (Error)",
            description=f"`{err_msg}`",
            color=0xEF4444,
            timestamp=datetime.now(timezone.utc),
        )
        embed.set_footer(text=f"⚡ System 1 (Jev) | {decision_ms}ms | Cost: $0")
        return embed

    containers = data.get("containers", [])
    color = 0x22C55E if containers else 0x6B7280
    embed = discord.Embed(
        title=f"🐳 Docker Containers ({len(containers)})",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )
    if not containers:
        embed.description = "No active containers found."
    else:
        for c in containers[:8]:
            status = c.get("status", "Unknown")
            icon = "🟢" if "Up" in status else "🔴"
            embed.add_field(
                name=f"{icon} {c.get('names', 'N/A')}",
                value=f"Status: `{status}`\nImage: `{c.get('image', 'N/A')}`",
                inline=False,
            )
    embed.set_footer(text=f"⚡ System 1 (Jev) | {decision_ms}ms | Cost: $0")
    return embed


def create_git_embed(data: dict, decision_ms: float) -> discord.Embed:
    """Create a Git pull execution summary card."""
    is_success = data.get("status") == "success"
    color = 0x22C55E if is_success else 0xEF4444
    title = "🐙 Git Pull: Success" if is_success else "🐙 Git Pull: Failed"
    output_text = data.get("output", data.get("error", "No output"))
    embed = discord.Embed(
        title=title,
        description=f"```\n{output_text[:1500]}\n```",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )
    embed.set_footer(text=f"⚡ System 1 (Jev) | {decision_ms}ms | Cost: $0")
    return embed


# =============================================================================
# Bot Events
# =============================================================================

@bot.event
async def on_ready():
    print(f"🤖 Logged in as {bot.user.name} ({bot.user.id})")
    print(f"⚡ Dual-Process AI Bot (Jev × Gemini) running...")
    await bot.change_presence(
        activity=discord.Activity(type=discord.ActivityType.watching, name="Dual-Process AI")
    )


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    content = message.content.strip()
    if not content:
        return

    channel_id = message.channel.id
    history = list(conversation_memories[channel_id])

    # Help command
    if content.lower() in ["!help", "help", "ヘルプ"]:
        embed = discord.Embed(
            title="⚡ Dual-Process AI Bot",
            description=(
                "System 1 (Jev) instantly classifies and executes routine operations ($0).\n"
                "System 2 (Gemini 3.8 Flash) activates only for deep thinking."
            ),
            color=0x2563EB,
        )
        embed.add_field(
            name="⚡ System 1 (Instant Operations, $0)",
            value=(
                "• サーバー状態: `status` / `スペック`\n"
                "• Docker確認: `docker` / `コンテナ一覧`\n"
                "• Docker再起動: `docker restart <name>`\n"
                "• Git pull: `git pull` / `コード最新化`"
            ),
            inline=False,
        )
        embed.add_field(
            name="🧠 System 2 (Deep Reasoning)",
            value="• アーキテクチャ設計\n• コードレビュー\n• 複雑な課題の思考と生成",
            inline=False,
        )
        embed.set_footer(text="⚡ Dual-Process Pattern: Jev + Gemini")
        await message.reply(embed=embed)
        return

    # Process through dual-process router
    async with message.channel.typing():
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, router.process, content, history)

    decision = result["decision"]
    action = decision["action"]
    s2_engaged = result["system2_engaged"]
    j_ms = decision["latency_ms"]
    total_ms = result["total_latency_ms"]

    # Try to render structured data as embed
    if action == "get_status":
        try:
            data = json.loads(result["output"])
            embed = create_status_embed(data, j_ms)
            await message.reply(embed=embed)
            conversation_memories[channel_id].append({"role": "user", "text": content})
            conversation_memories[channel_id].append({"role": "model", "text": f"[Status] {result['output'][:200]}"})
            return
        except (json.JSONDecodeError, Exception):
            pass

    elif action == "docker_status":
        try:
            data = json.loads(result["output"])
            embed = create_docker_embed(data, j_ms)
            await message.reply(embed=embed)
            conversation_memories[channel_id].append({"role": "user", "text": content})
            conversation_memories[channel_id].append({"role": "model", "text": f"[Docker Status] {result['output'][:200]}"})
            return
        except (json.JSONDecodeError, Exception):
            pass

    elif action == "docker_restart":
        try:
            data = json.loads(result["output"])
            if data.get("status") == "success":
                embed = discord.Embed(
                    title="🔄 Docker Restart Success",
                    description=data.get("message", "Container restarted."),
                    color=0x22C55E,
                    timestamp=datetime.now(timezone.utc),
                )
            else:
                embed = discord.Embed(
                    title="❌ Docker Restart Failed",
                    description=f"`{data.get('error', 'Unknown error')}`",
                    color=0xEF4444,
                    timestamp=datetime.now(timezone.utc),
                )
            embed.set_footer(text=f"⚡ System 1 (Jev) | {j_ms}ms | Cost: $0")
            await message.reply(embed=embed)
            conversation_memories[channel_id].append({"role": "user", "text": content})
            conversation_memories[channel_id].append({"role": "model", "text": f"[Docker Restart] {result['output'][:200]}"})
            return
        except (json.JSONDecodeError, Exception):
            pass

    elif action == "git_pull":
        try:
            data = json.loads(result["output"])
            embed = create_git_embed(data, j_ms)
            await message.reply(embed=embed)
            conversation_memories[channel_id].append({"role": "user", "text": content})
            conversation_memories[channel_id].append({"role": "model", "text": f"[Git Pull] {result['output'][:200]}"})
            return
        except (json.JSONDecodeError, Exception):
            pass

    # Default: text response with system badge
    if s2_engaged:
        badge = f"🧠 **[System 2 / Gemini]** ({total_ms:.0f}ms)"
    else:
        badge = f"⚡ **[System 1 / Jev]** ({j_ms}ms / $0)"

    full_msg = f"{badge}\n\n{result['output']}"

    # Update memory
    conversation_memories[channel_id].append({"role": "user", "text": content})
    conversation_memories[channel_id].append({"role": "model", "text": result["output"]})

    # Send (split if over Discord's 2000 char limit)
    if len(full_msg) <= 1950:
        await message.reply(full_msg)
    else:
        chunks = [full_msg[i:i + 1900] for i in range(0, len(full_msg), 1900)]
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                await message.reply(chunk)
            else:
                await message.channel.send(chunk)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        print("Error: Set DISCORD_BOT_TOKEN in .env", file=sys.stderr)
        sys.exit(1)

    print("Starting Dual-Process AI Discord Bot...")
    bot.run(DISCORD_BOT_TOKEN)
