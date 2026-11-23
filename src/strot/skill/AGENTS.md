# STROT compiler

You can compile a specialized, $0 local **decision model** from a user's data —
for classify / route / triage work (tickets, claims, leads, issues) that would
otherwise cost an LLM call per record.

You are the teacher. STROT is a deterministic compiler: it holds no API key and
runs no LLM. You design the typed questions and label a small sample yourself;
STROT distills a calibrated local `.strot` model that decides every record after
at $0.

## Setup

```bash
pip install "strot-ai[local]"
```

## Method

The full methodology is in **PLAYBOOK.md** (alongside this file): inspect → design
typed questions → sample 25–100 records → **label them yourself** → compile →
verify → dispatch → route by confidence → monitor → recompile.

## Compile

Write your questions + labels to JSON, then run:

```bash
python scripts/compile_model.py --data data.json --questions questions.json \
  --labels labels.json --name triage
# -> triage.strot
```

Decide locally, for free:

```python
import strot
strot.load_model("triage")(record)   # typed JSON + calibrated confidence
```

Route by confidence: high → the local model; low / refusal → handle it yourself.
Each taken-over decision is an LLM call saved. Re-label + recompile as data grows.
