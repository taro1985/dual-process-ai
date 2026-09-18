# 🧠⚡ Dual-Process AI

**A design pattern that combines a near-zero-latency System 1 classifier with a deep-reasoning LLM, inspired by Daniel Kahneman's *Thinking, Fast and Slow*.**

> "Don't send everything to an LLM. Fast brain + deep brain."

---

## 💡 The Idea

Modern AI applications route **every** user request to a large language model — even trivial ones like "show server status" or "list repos". This wastes time, tokens, and money.

**Dual-Process AI** mirrors the human brain's two cognitive systems:

| | System 1 (Fast) | System 2 (Slow) |
|---|---|---|
| **Kahneman** | Intuitive, automatic | Deliberate, analytical |
| **AI Implementation** | [Jev](https://typesafe.ai) (TypeSafe AI) | Gemini 3.8 Flash |
| **Latency** | 0.01 ms | 500–5000 ms |
| **Cost** | $0 | Token-based |
| **Use case** | Routing, classification, safety gates | Reasoning, code generation, analysis |

```
User Input
    │
    ▼
┌──────────────────────┐
│  System 1 (Jev)      │  ← 0.01ms, $0
│  Route / Classify    │
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     │           │
  Routine    Thinking
  Task       Required
     │           │
     ▼           ▼
┌─────────┐ ┌──────────────────┐
│ Direct  │ │  System 2 (LLM)  │  ← Deep reasoning
│ Execute │ │  Gemini 3.8 Flash│
└─────────┘ └──────────────────┘
```

---

## 📦 What's Included

| File | Description |
|---|---|
| `router.py` | Core Dual-Process router — Jev classifies, Gemini reasons |
| `safety_gate.py` | PreToolUse hook for AI coding agents (e.g., Antigravity) — blocks dangerous commands in 0.01ms |
| `discord_bot.py` | Discord bot with rich embed UI and short-term context memory |
| `.env.example` | Template for API keys |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install google-genai discord.py python-dotenv
# Optional: pip install typesafe-sdk  (for Jev API mode)
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the router

```bash
python router.py
```

### 4. Run the Discord bot

```bash
python discord_bot.py
```

### 5. Use the safety gate (Antigravity / AI coding agents)

Add to your hooks configuration:

```json
{
  "hooks": [
    {
      "event": "PreToolUse",
      "command": "python /path/to/safety_gate.py",
      "timeout_ms": 5000
    }
  ]
}
```

---

## 🏗️ Architecture Deep Dive

### Router (`router.py`)

The `DualProcessRouter` has two layers:

1. **`JevClassifier`** — Instant intent classification using keyword matching (fallback) or [TypeSafe Jev API](https://typesafe.ai) (when API key is available). Returns an action and whether deep reasoning is needed.

2. **`GeminiReasoner`** — Activated *only* when System 1 says `needs_reasoning = True`. Supports conversation history for multi-turn context.

```python
from router import DualProcessRouter

router = DualProcessRouter()
result = router.process("What's the server status?")
# → System 1 handles it instantly, no LLM call

result = router.process("Design a microservice architecture for this")
# → System 1 routes to System 2, Gemini generates a thoughtful response
```

### Safety Gate (`safety_gate.py`)

A reflexive guard that inspects shell commands **before** an AI agent executes them:

```
rm -rf /        → ⚡ DENY (0.01ms)
DROP DATABASE   → ⚡ DENY (0.01ms)
fork bomb       → ⚡ DENY (0.01ms)
echo hello      → ✅ ALLOW
git status      → ✅ ALLOW
```

Works as a standalone script or with the TypeSafe Jev API for enhanced detection.

### Discord Bot (`discord_bot.py`)

A smartphone-optimized Discord bot with:
- **Rich embed cards** — Color-coded server status, GitHub repos, Drive files
- **Short-term context memory** — Remembers recent turns per channel
- **Dual badge display** — Shows which system handled each request

---

## 🔧 Configuration

### Without Jev API (Keyword Fallback)

Works out of the box. System 1 uses lightweight keyword matching for routing. Zero external dependencies beyond Gemini.

### With Jev API (Full System 1)

1. Get an API key from [TypeSafe AI Console](https://console.typesafe.ai)
2. Add `TYPESAFE_API_KEY=your_key` to `.env`
3. Install: `pip install typesafe-sdk`

The router automatically detects and uses the Jev API when available.

---

## 🎯 Design Principles

1. **LLM as last resort** — Don't invoke a billion-parameter model for a lookup table task
2. **Fail-open** — If System 1 can't decide, escalate to System 2 (never block)
3. **Cost-aware** — Track and display which system handled each request
4. **Context-preserving** — System 2 receives conversation history for coherent multi-turn dialogue
5. **Reflexive safety** — Dangerous commands are caught before the LLM even thinks

---

## 📊 Performance

Measured on a Sony VAIO (2-core, 3.7GB RAM):

| Metric | System 1 Only | System 2 Engaged |
|---|---|---|
| Latency | 0.01–0.05 ms | 500–5000 ms |
| API Cost | $0 | Token-based |
| RAM | ~50 MB | ~50 MB (API call) |

---

## 🤝 Credits

- [Kahneman, D. (2011). *Thinking, Fast and Slow*](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) — The cognitive science inspiration
- [TypeSafe AI / Jev](https://typesafe.ai) — System 1 non-autoregressive classifier
- [Google Gemini](https://ai.google.dev) — System 2 deep reasoning

---

## 📄 License

MIT
