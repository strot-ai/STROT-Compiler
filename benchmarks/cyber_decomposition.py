"""Experiment: does compile-time DECOMPOSITION move the cyber 39.5% without
fine-tuning the model?

Hypothesis: a frozen 144M model does several easy semantic splits better than
one hard 6-way decision. The compiler's job (here hand-designed as a proxy for
what the frontier LLM would synthesize) is to find that decomposition.

Same real VCDB held-out test set for both arms. Real ground-truth labels are the
ONLY judge (no LLM grading). Honest confound: decomposition compounds errors, so
it is NOT guaranteed to win -- that is exactly what this measures.

  FLAT         strot.run on the 6-way registry (reproduces the ~39.5% number)
  DECOMPOSED   stage A (coarse 3-way) -> stage B (hacking/malware) or
               stage C (social/physical/misuse); accidental -> error

Per-stage accuracy is printed too, to see where decomposition helps or compounds.
"""

import collections
import io
import json
import time
import urllib.request
import zipfile

import strot
from julia import load_model

engine = load_model("/opt/julia-1", device="cpu")
VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
LLM = "anthropic/claude-opus-4-8"
CAP = 25
SAMPLE_PER = 6
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]

FINE_DESC = {
    "hacking": "Intrusion via technical means: stolen/brute-forced credentials, exploited vulnerabilities, SQL injection, remote access.",
    "malware": "Malicious software: ransomware, spyware, trojans, keyloggers, worms installed on a system.",
    "error": "Unintentional mistake: misdelivery, misconfiguration, lost device, accidental publishing or disposal of data.",
    "misuse": "Authorized insider abusing access: snooping, data theft by staff, policy violation.",
    "physical": "Physical action: theft of a device/document, tampering, surveillance, break-in.",
    "social": "Social engineering of a person: phishing, pretexting, bribery, deception.",
}


def _get(url, timeout=90):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(8 * (attempt + 1))


def balance(items, label_of, text_of, classes, cap, sample_per):
    buckets = collections.defaultdict(list)
    for it in items:
        lab = label_of(it)
        if lab in classes and isinstance(text_of(it), str) and len(text_of(it).strip()) > 40:
            if len(buckets[lab]) < cap:
                buckets[lab].append(text_of(it).strip())
    s_texts, t_texts, t_truth = [], [], []
    for c in classes:
        rows = buckets[c]
        s_texts.append((c, rows[:sample_per]))
        for r in rows[sample_per:]:
            t_texts.append(r)
            t_truth.append(c)
    return dict(s_texts), t_texts, t_truth, {c: len(buckets[c]) for c in classes}


def run_stage(name, sample_texts, records, registry, query, question):
    """Compile+run one STROT decision. Returns {record_index: predicted_label}."""
    sample = [{"text": t[:1500]} for t in sample_texts]
    recs = [{"text": t[:1500]} for t in records]
    off = len(sample)
    result = strot.run(sample + recs, query, registry, question=question,
                       measure_labels=True, llm=LLM, state=f"/tmp/dec_{name}",
                       sample_size=len(sample))
    routed = {o["index"]: o["action"] for o in result.outcomes}
    return {i: routed.get(i + off) for i in range(len(records))}


def main():
    data = json.loads(zipfile.ZipFile(io.BytesIO(_get(VCDB_ZIP))).read("vcdb_1-of-1.json"))

    def label(r):
        acts = [a for a in r.get("action", {}) if a in CLASSES]
        return acts[0] if len(acts) == 1 else None

    samples, test, truth, counts = balance(data, label, lambda r: r.get("summary"),
                                           CLASSES, CAP, SAMPLE_PER)
    n = len(test)
    rnd = round(100 / 6, 1)
    maj = round(collections.Counter(truth).most_common(1)[0][1] / n * 100, 1)
    print(f"classes: {counts}")
    print(f"test records: {n}   random: {rnd}%   majority: {maj}%\n", flush=True)

    # ---------- FLAT 6-way ----------
    flat_sample = [t for c in CLASSES for t in samples[c]]
    flat_pred = run_stage("flat", flat_sample, test, FINE_DESC,
                          "Classify each security-incident report by its primary attack action.",
                          "What was the primary attack action in this security incident?")
    flat_correct = sum(1 for i in range(n) if flat_pred.get(i) == truth[i])
    print(f"FLAT 6-way compiled:  {round(flat_correct / n * 100, 1)}%", flush=True)

    # ---------- DECOMPOSED ----------
    # Stage A: coarse 3-way
    COARSE = {
        "accidental": "An unintentional mistake with no attacker: misdelivery, misconfiguration, a lost device, accidental publishing or disposal of data.",
        "digital_attack": "A deliberate technical/software attack on systems: hacking, stolen credentials, exploited vulnerabilities, malware, ransomware.",
        "human_or_physical": "A deliberate act involving a person or a physical object: social engineering/phishing, insider misuse of access, or physical theft/tampering.",
    }
    coarse_of = {"error": "accidental", "hacking": "digital_attack", "malware": "digital_attack",
                 "social": "human_or_physical", "misuse": "human_or_physical", "physical": "human_or_physical"}
    a_sample = flat_sample
    a_pred = run_stage("A", a_sample, test, COARSE,
                       "Sort each security incident into one of three coarse buckets.",
                       "Is this incident an accidental mistake, a digital attack, or a human/physical act?")

    # Stage B: hacking vs malware (digital subset)
    b_sample = samples["hacking"] + samples["malware"]
    B_DESC = {k: FINE_DESC[k] for k in ("hacking", "malware")}
    digital_idx = [i for i in range(n) if a_pred.get(i) == "digital_attack"]
    b_pred = run_stage("B", b_sample, [test[i] for i in digital_idx], B_DESC,
                       "Classify each digital-attack incident.",
                       "Was this primarily a hacking intrusion or malware?") if digital_idx else {}
    b_map = {digital_idx[j]: b_pred.get(j) for j in range(len(digital_idx))}

    # Stage C: social vs misuse vs physical (human/physical subset)
    c_sample = samples["social"] + samples["misuse"] + samples["physical"]
    C_DESC = {k: FINE_DESC[k] for k in ("social", "misuse", "physical")}
    hp_idx = [i for i in range(n) if a_pred.get(i) == "human_or_physical"]
    c_pred = run_stage("C", c_sample, [test[i] for i in hp_idx], C_DESC,
                       "Classify each human-or-physical incident.",
                       "Was this social engineering, insider misuse, or a physical act?") if hp_idx else {}
    c_map = {hp_idx[j]: c_pred.get(j) for j in range(len(hp_idx))}

    # Compose
    dec_pred = {}
    for i in range(n):
        a = a_pred.get(i)
        if a == "accidental":
            dec_pred[i] = "error"
        elif a == "digital_attack":
            dec_pred[i] = b_map.get(i)
        elif a == "human_or_physical":
            dec_pred[i] = c_map.get(i)
        else:
            dec_pred[i] = None
    dec_correct = sum(1 for i in range(n) if dec_pred.get(i) == truth[i])

    # Stage A accuracy (coarse)
    a_correct = sum(1 for i in range(n) if a_pred.get(i) == coarse_of[truth[i]])
    print(f"DECOMPOSED compiled:  {round(dec_correct / n * 100, 1)}%", flush=True)
    print(f"\n  stage A (coarse 3-way): {round(a_correct / n * 100, 1)}%  "
          f"(digital={len(digital_idx)}, human/phys={len(hp_idx)})")
    if digital_idx:
        b_c = sum(1 for i in digital_idx if b_map.get(i) == truth[i] and truth[i] in ("hacking", "malware"))
        print(f"  stage B (hacking/malware) on {len(digital_idx)} routed: {round(b_c / len(digital_idx) * 100, 1)}% fine-correct")
    if hp_idx:
        c_c = sum(1 for i in hp_idx if c_map.get(i) == truth[i] and truth[i] in ("social", "misuse", "physical"))
        print(f"  stage C (social/misuse/physical) on {len(hp_idx)} routed: {round(c_c / len(hp_idx) * 100, 1)}% fine-correct")


main()
