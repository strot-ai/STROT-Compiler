# Benchmarks

Reproducible scripts for the behaviors described in the main README. These are
the scripts committed to the repo; each runs end to end and prints its own
baselines so you can read the results against chance and majority-class.

Accuracy numbers are deliberately not reported here — run the scripts on your
own environment. The point of each script is the *shape* of the result
(baselines, where compilation helps, where it does not), not a headline figure.

## Routing: `routing_mpg.py`

A grounded-routing benchmark on
[seaborn's `mpg.csv`](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv)
— cars with a ground-truth `origin` column (usa / japan / europe).

```bash
curl -sLO https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv
pip install strot-ai[local,anthropic]
python -c "import strot; strot.prefetch_nlu(local_dir='./julia-1', install=True)"
export STROT_NLU_CHECKPOINT=./julia-1 ANTHROPIC_API_KEY=...
python routing_mpg.py
```

Route each car to one of three market workflows by origin. Two passes:

- **Pass A** — `origin` present in the record (grounded choice; ground truth
  known per record).
- **Pass B** — `origin` stripped (semantic inference; documents the small
  decision model's known weakness — do not use this mode in production).

Notes, read before quoting anything:

- The compiler **never sees the ground-truth labels**. Routing labels and the
  state projection are synthesized from the action-registry descriptions and the
  data schema alone; accuracy is then measured against the held-out `origin`
  column.
- **Compile variance is real**: synthesis is stochastic, so repeat runs produce
  different artifacts with different accuracy. The golden-record gate (`golden=`)
  exists precisely to bound this — production use should always pin known
  answers.

## Semantic decisions over unstructured text: `semantic_text.py`

Does compilation help when the answer is **not** a structured field? Two public
datasets with their own ground-truth labels (AG News, 4-class news topic; DAIR
Emotion, 6-class tweets); a held-out test split each; the compiler never sees
any test labels. Three conditions: naive (raw text + bare class names into the
decision model), STROT-compiled, and compiled + one labeled train example per
class as goldens.

Honest findings, including where STROT loses:

1. Compilation delivers large gains when integration is the bottleneck (AG News:
   the synthesized labels/projection beat bare class names on text the model must
   genuinely read).
2. Compilation can *hurt* when the naive interface is already optimal (Emotion
   tweets literally contain the class words, so bare class names are near-perfect
   labels and synthesized wording dilutes them). Measured label search
   (`measure_labels=True`) fixes this: the compiler scores synthesized labels vs
   bare registry keys on the teacher-labeled sample and bakes the winner, so a
   weak synthesized labeling cannot regress below bare class names. It bounds the
   downside but does not eliminate synthesis variance, and a small sample is a
   noisy judge — multiple synthesis candidates and larger measurement samples are
   the known next steps.
3. Using goldens as an accuracy optimizer backfires: gate-driven retries select
   wording that satisfies a handful of examples but generalizes worse. Goldens
   are a **regression gate** against broken or variant compiles — not an
   optimization signal. Use clear, unambiguous goldens; don't tune accuracy with
   them.

## Real risk narratives: `risk_narratives.py`

Serious, record-level, unstructured text — the shape cyber, financial, and
insurance triage actually run on. Two fully public, ground-truth-labeled tasks,
both categorical triage over free narrative:

- **Cyber** — [VERIS Community Database](https://github.com/vz-risk/VCDB) (the
  Verizon DBIR breach corpus): the incident `summary` narrative → primary attack
  action, 6 classes (hacking / malware / error / misuse / physical / social).
  Single-action incidents only; the combined `vcdb.json.zip` is fetched directly.
- **Finance** — [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/):
  the first-person `consumer_narrative` → product, 6 classes (credit reporting /
  debt collection / credit card / bank account / mortgage / money transfer).

```bash
pip install strot-ai[local,anthropic]
python -c "import strot; strot.prefetch_nlu(local_dir='./julia-1', install=True)"
export STROT_NLU_CHECKPOINT=./julia-1 ANTHROPIC_API_KEY=...
python risk_narratives.py
```

Classes are **balanced** so accuracy is interpretable; every task prints its
random and majority-class baselines. `measure_labels=True` is on, so a weak
synthesized labeling cannot regress below bare class names. The compiler never
sees the test labels.

Honest findings:

1. **Both clear their baselines and the compiler lifts the same model.** This is
   the credible STROT story on messy real text: measured integration +
   calibration on a frozen local model.
2. **These are proof the architecture works on real narratives, not production
   accuracy.** Above chance but far from an underwriter's bar. The cyber taxonomy
   is genuinely ambiguous at the record level (a stolen laptop is `physical` or
   `error`; phishing-into-intrusion blurs `social` and `hacking`), which caps a
   generalist encoder.
3. **Calibration abstains on the uncertain tail** rather than guessing — visible
   honesty, not a hidden accuracy tax.
4. **The fix for production accuracy is the model slot, not the compiler.** Drop
   a task-trained specialist into `nlu=`; the synthesis, calibration, and gate
   are unchanged.

## Input-formatting ablation: `ablation_formats.py`

Same model, same records, same question — only state/option formatting varies.
This isolates how much of routing accuracy is integration rather than model
capability. Runs the decision model directly; no LLM required.
