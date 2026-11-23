---
name: strot-watcher
description: >-
  Learn a $0 local decision model by watching another agent decide, then take
  over its routine calls. Use when an agent (yours or one you observe) makes the
  same typed decision over and over — routing, triage, yes/no flags — and you want
  those decided locally for free, deferring only the hard ones back to the agent.
  No labels needed: the observed agent's decisions ARE the labels.
---

# STROT watcher — learn an agent's routine decisions, then take them over

An agent's `(input → decision)` pairs are **labels** — real, free, and
in-distribution. The watcher distills them into a local **understudy** that takes
the confident cases at $0 and defers the rest back to the agent. The share it
handles grows as it learns — a gradual, self-shifting takeover, never a hard
cutover, never a lost decision.

> It clones the agent's **policy**, not truth (fidelity ≠ correctness), and only
> for **typed/discrete** decisions (predicate / choice) — not free-form text.

## Setup

```bash
pip install "strot-ai[local]"
```

## The loop

1. **OBSERVE** — collect the agent's decision traces. Any of:
   - a JSONL / JSON file of `{"input": {record}, "decision": {question: answer}}`,
   - **OpenTelemetry spans** whose attributes carry `input` and `decision`
     (stringified JSON is decoded).
2. **INGEST** — `records, labels = strot.ingest_traces(traces)`. The decisions
   become the labels. (Pass `input_key=` / `decision_key=` if your spans use other
   attribute names.)
3. **RECOVER THE SCHEMA** — `questions = strot.infer_questions(labels)`:
   all-boolean keys → `predicate`, others → `choice` over the observed values.
   Review it; add instructions/descriptions if they help.
4. **COMPILE the understudy** — `model = strot.compile(records, questions, labels)`
   (`mode="deep"` for long/representation-bound text). No LLM.
5. **SHADOW** — run the understudy alongside the agent on held-out traces and
   `strot.evaluate(model, labeled)` for **fidelity** (does it agree with the
   agent?) + calibration. Do not take over until fidelity and confidence clear
   your bar.
6. **TAKE OVER, confidence-gated** — in the harness, call the understudy **before**
   the agent: high confidence → use the $0 local decision; low confidence or
   `{"refusal": true, "reason": "ood"}` → defer to the agent. Every taken-over
   call is an agent/LLM call saved. (Gate the FIRST takeover on human approval.)
7. **MONITOR & RE-LEARN** — log decisions (`model.trace(...)`), watch
   `strot.drift_check(traces)`; when the agent's behavior shifts, ingest fresh
   traces and `recompile`. If drift is bad, pause the understudy and hand all
   traffic back to the agent until it relearns.

## Principles

- **The decisions are the labels.** No teacher, no LLM, no key — you distill what
  the agent already did.
- **Confidence is the router**, and the split shifts over time: as the understudy
  learns more of the agent's behavior, more traffic flows to the $0 model
  automatically.
- **Typed decisions only.** Classify / route / yes-no / tool-choice — never
  free-form generation. Fidelity to the agent is capped at the agent's own quality.
- **Fallback on uncertainty; first takeover human-gated; drift-monitored** — the
  agent is always the safety net.
