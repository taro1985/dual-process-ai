# Reddit Post (URL-Free Filter Bypass Edition)

**Target Subreddit:** [r/google_antigravity](https://www.reddit.com/r/google_antigravity/submit)

---

## 📌 Title (タイトル)

```text
Built a 0.01ms PreToolUse safety gate for Antigravity using Jev (TypeSafe AI) + Dual-Process AI pattern
```

---

## 📝 Body (本文 - この下の枠内を最初から最後まで丸ごとコピー)

````markdown
Hey everyone,

I built an open-source implementation of what I'm calling **Dual-Process AI** — applying Daniel Kahneman's *Thinking, Fast and Slow* to AI systems.

(GitHub repo: **taro1985/dual-process-ai**)

### The Problem
Most AI applications route every single request to a massive LLM — even trivial lookup tasks like "show server status" or "list repos". It's slow, expensive, and wastes compute.

### The Solution
We split the workload into two distinct cognitive layers:
- **System 1 (Fast & Intuitive):** Uses Jev by TypeSafe AI (a non-autoregressive classifier trained with RLCD for calibrated confidence). Latency: sub-millisecond to ~70ms. Cost: ~$0.
- **System 2 (Slow & Deliberate):** Uses Gemini 3.8 Flash for deep reasoning, multi-turn architecture design, and code generation.

### Key Mechanism: Calibrated Confidence, Not Just Speed
Instead of asking an RLHF-tuned LLM "how confident are you?" (which produces hallucinations shaped by human preference), Jev outputs calibrated probability scores you can safely put in an `if` statement:

```python
confidence >= 0.85  → System 1 handles directly (0.01ms / $0)
confidence <  0.85  → Escalate to System 2 (Gemini)
```

### Included in the Repo:
- **`router.py`**: Confidence-threshold routing engine with graceful degraded fallback for testing without an API key.
- **`safety_gate.py`**: Reflexive guard for agentic coding tools (e.g., Google Antigravity PreToolUse hook) — catches catastrophic commands like `rm -rf /` in 0.01ms before the LLM can execute them.
- **`discord_bot.py`**: Smartphone-optimized Discord bot with per-channel context memory.

I tested this running on an old Sony VAIO server (2 cores, 3.7GB RAM) to prove the fast path needs zero local GPU resources.

Feedback, critiques, and ideas on calibration metrics (Brier score evaluation) are very welcome!
````
