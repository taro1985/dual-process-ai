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
from dotenv import load_dotenv

import discord
from discord.ext import commands

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

from router import DualProcessRouter

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
# Example Tool Handlers (replace with your own)
# =============================================================================

def get_status(_input: str) -> str:
    """Replace this with your actual status check logic."""
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


# Register tool handlers
TOOL_HANDLERS = {
    "get_status": get_status,
    # Add more: "list_repos": your_github_func, etc.
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
                "System 1 (Jev) instantly classifies your request.\n"
                "System 2 (Gemini) activates only for deep thinking."
            ),
            color=0x2563EB,
        )
        embed.add_field(
            name="⚡ System 1 (instant, $0)",
            value="• Server status\n• GitHub repos\n• Google Drive files",
            inline=False,
        )
        embed.add_field(
            name="🧠 System 2 (deep thinking)",
            value="• Architecture design\n• Code review\n• Creative tasks",
            inline=False,
        )
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
