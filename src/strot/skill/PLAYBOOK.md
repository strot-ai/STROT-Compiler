# STROT — compile a decision model

**What STROT is:** you (the agent) reason once over a small sample of the user's
data; STROT distills that judgment into a specialized **decision model** — a
calibrated head on a frozen encoder — that then decides *every* record locally,
on CPU, at **$0**. **You are the teacher.** STROT runs no LLM and holds no API
key: it only embeds, fits the calibrated heads + an out-of-distribution gate, and
bakes a self-contained `.strot` model.

> Reason once, at compile time. Decide forever, locally.

You bring the intelligence (designing the questions, labeling the sample). STROT
is the deterministic compiler. One install — `pip install "strot-ai[local]"` —
and no keys.

## The loop

**1. INSPECT** — look at the data: fields, types, cardinality, a few sample
rows. If the decision hinges on raw numbers (amounts, codes), note it: the
encoder is text, weak on bare numbers — prefer text-driven decisions.

**2. DESIGN the typed questions** — turn the user's intent into a small set of
typed questions. Three types:
- `predicate` — a true/false statement (returns P(true)).
- `choice` — one of N named options.
- `score` — one of N ordered levels.

Keep each question discrete and answerable from the record's text. Example:

```json
{
  "is_bug_report": {"type": "predicate", "instructions": "mentions a crash or error?"},
  "team": {"type": "choice", "choices": [
    {"value": "billing", "description": "charges, invoices, refunds"},
    {"value": "engineering", "description": "crashes, bugs, defects"}]}
}
```

**3. SAMPLE** — take a small sample (25–100 records is plenty; a calibrated head
needs only a few examples per class). You label this once; the model handles
everything after.

**4. LABEL the sample — this is your job as the teacher.** For each sampled
record, answer every question yourself (that's what your LLM is for). Produce one
`{question: answer}` dict per record, in the same order as the data:

```json
[
  {"is_bug_report": true,  "team": "engineering"},
  {"is_bug_report": false, "team": "billing"}
]
```

Cover each outcome with a few records — a head can only learn an answer it has
seen. If PII is a concern: STROT sends records nowhere (it embeds locally), but
*you* read them to label — so if you're a hosted agent, run `strot.pii_scan(data)`
to see what structured PII a record carries, and prefer a local/open-weight agent
when data can't leave your infra.

**5. COMPILE** — hand your labels to STROT; it distills the model (no LLM, no
key). Run the bundled script, or call the library directly:

```python
import strot
model = strot.compile(data, questions, labels)   # -> a DecisionModel
model.dispatch("triage")                          # -> triage.strot
```

**6. VERIFY** — hold out some labeled records and score the model:

```python
report = strot.evaluate(model, held_out)   # accuracy / coverage / ECE per question
```

If accuracy is low: label more real records (especially for confused classes),
or try `mode="deep"` (the LoRA tier) for long/representation-bound text. Labels
you trust are the best input — human ground truth beats your own labels.

**7. DISPATCH & ROUTE** — the compiled model is a callable, self-contained file:

```python
same = strot.load_model("triage")
decision = same(record)     # typed JSON + a calibrated confidence
```

Route by that confidence: **high confidence → trust the $0 local model; low
confidence / `{"refusal": true, "reason": "ood"}` → fall back to yourself (the
agent).** Every record the model takes over is an LLM call saved. As you label
more, the confidently-handled share grows — a gradual takeover, never a hard
cutover, never a lost decision.

**8. MONITOR & RECOMPILE** — log decisions with `model.trace("traces.jsonl")`,
then watch them:
- `strot.read_traces(path)` — volume, refusal rate, confidence, value mix.
- `strot.drift_check(traces)` — recent-vs-baseline drift + a recompile verdict.
- `strot.cluster_rejections(traces)` — what's being refused (the gap to close).

When drift recommends it: gather more real records of the gap, **re-label**, and
recompile. Every compile is a fresh fit on the current data — never a warm-start
(so retraining can only improve; it's immune to forgetting).

## Principles

- **You are the teacher; STROT is deterministic.** Every AI step (design, label)
  is yours. STROT embeds, fits, bakes — no LLM, no key, no network.
- **Labels from any source.** Your own decisions, or human ground truth (the
  strongest). You always supply them; STROT never invents data.
- **Recompile, don't increment.** Improve by gathering more real data and
  recompiling, not by nudging weights.
- **The model refuses what it shouldn't decide.** Low-confidence / out-of-
  distribution records come back as refusals by design — escalate those to
  yourself and learn from them.
