---
title: STROT-AI Compiler
emoji: 🧪
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Compile SYS 2 reasoning into SYS 1 decisions, $0 inference
---

# STROT — the open decision system

**Compile System 2 reasoning into System 1 decisions — $0 at inference.**

**LOW LATENCY · ULTRAFAST · CHEAP & FREE · 0 TOKENS · LOW CARBON · CPU INFERENCE ONLY**

An LLM reasons through your task once, at compile time. Every decision after runs
on a small local model (SupersonicLabs/Julia-1, 144M) on CPU — no LLM call, no
API key: faster, cheaper, and a fraction of the energy of per-record LLM calls.
Output is bounded typed-JSON with confidence; low-confidence records abstain.

- **Try it** runs records through pre-compiled artifacts. This is the runtime:
  key-free, CPU-only, $0 per decision.
- **Compile your own** uses your API key once to build a fresh artifact.

Source: https://github.com/strot-ai/STROT-Compiler · `pip install strot-ai`
