"""Path 3 measurement: STROT as a cost-optimizing triage gate on cyber.

Run the default model (Julia) with conformal calibration at target error rates,
and measure the REAL operating point -- replacing the guessed "45% local / 55%
escalate" with facts:

  answered  = dispatched locally ($0, CPU)           -> result.outcomes
  escalated = abstained, sent to review/cloud LLM     -> result.fallbacks
  acc@answered = accuracy on the answered slice
  wrong-dispatch rate = wrong dispatched / ALL records  (must be <= alpha if the
                        marginal guarantee holds)

Honest note: Julia is the weak default (~40-50% zero-shot on cyber), so expect a
high escalation rate. A stronger base model would widen the answered slice; this
is the floor, measured.
"""

import collections
import io
import json
import time
import urllib.request
import zipfile

import strot
from julia import load_model

load_model("/opt/julia-1", device="cpu")
VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
LLM = "anthropic/claude-opus-4-8"
CAP, SAMPLE_PER = 45, 24
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
REG = {
    "hacking": "Intrusion via technical means: stolen/brute-forced credentials, exploited vulnerabilities, SQL injection, remote access.",
    "malware": "Malicious software: ransomware, spyware, trojans, keyloggers, worms installed on a system.",
    "error": "Unintentional mistake: misdelivery, misconfiguration, lost device, accidental publishing or disposal of data.",
    "misuse": "Authorized insider abusing access: snooping, data theft by staff, policy violation.",
    "physical": "Physical action: theft of a device/document, tampering, surveillance, break-in.",
    "social": "Social engineering of a person: phishing, pretexting, bribery, deception.",
}
Q = "What was the primary attack action in this security incident?"
GOAL = "Classify each security-incident report by its primary attack action."


def _get(url, timeout=120):
    for a in range(5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == 4:
                raise
            time.sleep(8 * (a + 1))


def main():
    data = json.loads(zipfile.ZipFile(io.BytesIO(_get(VCDB_ZIP))).read("vcdb_1-of-1.json"))

    def label(r):
        acts = [a for a in r.get("action", {}) if a in CLASSES]
        return acts[0] if len(acts) == 1 else None

    buckets = collections.defaultdict(list)
    for r in data:
        lab = label(r)
        s = r.get("summary")
        if lab in CLASSES and isinstance(s, str) and len(s.strip()) > 40 and len(buckets[lab]) < CAP:
            buckets[lab].append(s.strip())
    sample, test, truth = [], [], []
    for c in CLASSES:
        rows = buckets[c]
        sample += rows[:SAMPLE_PER]
        for t in rows[SAMPLE_PER:]:
            test.append(t); truth.append(c)
    n = len(test)
    idx_of = {c: i for i, c in enumerate(CLASSES)}
    print(f"test records: {n}   calibration sample: {len(sample)}\n", flush=True)

    recs = [{"text": t[:1500]} for t in sample] + [{"text": t[:1500]} for t in test]
    off = len(sample)

    for alpha in (0.10, 0.05):
        res = strot.run(recs, GOAL, REG, question=Q, measure_labels=True,
                        target_risk=alpha, llm=LLM, state=f"/tmp/p3_{int(alpha*100)}",
                        sample_size=off)
        routed = {o["index"]: o["action"] for o in res.outcomes}
        answered = [i for i in range(n) if routed.get(i + off) is not None]
        correct = sum(1 for i in answered if idx_of[routed[i + off]] == idx_of[truth[i]])
        a = len(answered)
        wrong_all = (a - correct) / n * 100  # the marginal guarantee target
        print(f"=== target error alpha = {alpha} ===")
        print(f"  answered locally : {a}/{n}  ({round(a/n*100,1)}%)   $0 CPU")
        print(f"  escalated        : {n-a}/{n}  ({round((n-a)/n*100,1)}%)   -> review/LLM")
        print(f"  accuracy @ answered: {round(correct/a*100,1)}%" if a else "  (answered 0)")
        print(f"  wrong-dispatch / ALL: {round(wrong_all,1)}%   (guarantee: <= {int(alpha*100)}%)  "
              f"{'HELD' if wrong_all <= alpha*100 + 1e-9 else 'EXCEEDED'}\n", flush=True)


main()
