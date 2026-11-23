---
name: strot-compiler
description: >-
  Compile a specialized, $0 local decision model from a user's data. Use when the
  user wants to classify / route / triage records at scale (support tickets,
  claims, leads, issues) and would otherwise call an LLM per record. You design
  typed questions, label a small sample yourself, and STROT distills a calibrated
  local model that decides every record on CPU for free. No API key.
---

# STROT compiler

You are the teacher. STROT is a deterministic compiler — it holds no API key and
runs no LLM; you supply the questions and the labels, it distills a local
`.strot` decision model that decides every record after at $0.

## Setup (once)

```bash
pip install "strot-ai[local]"
```

## Method

Follow **PLAYBOOK.md** in this skill directory — it is the full methodology
(inspect → design questions → sample → **label the sample yourself** → compile →
verify → dispatch → route by confidence → monitor → recompile).

## The one script you run

After you've designed the questions and labeled the sample, write them to JSON and
run the bundled helper:

```bash
python scripts/compile_model.py \
  --data data.json \
  --questions questions.json \
  --labels labels.json \
  --name triage
# -> writes triage.strot and prints its path
```

- `data.json` — the records (a JSON list of objects).
- `questions.json` — your typed-question config (see PLAYBOOK.md).
- `labels.json` — one `{question: answer}` dict per record, same order as data.
- Pass `--eval-labels held_out.json` to also print an accuracy/coverage/ECE report.

Then decide locally, for free:

```python
import strot
model = strot.load_model("triage")
model(record)   # typed decision JSON + a calibrated confidence
```

Route by the confidence: high → trust the local model; low / `{"refusal": true}`
→ handle it yourself. Every record the model takes over is an LLM call saved.
