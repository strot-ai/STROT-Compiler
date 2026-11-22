"""Probe: does head distillation (train a head on frozen encoder embeddings)
move cyber? Same real VCDB held-out test set, real labels as judge.

  no-distill   strot.run(...)                 (zero-shot Julia, reproduces ~40-46%)
  distill      strot.run(..., distill=True)    (teacher-labeled head on frozen
                                                embeddings, kept only if it beats
                                                zero-shot on a held-out fold)

distill=True gives max(zero-shot, head) by STROT's own internal real-fold check,
so it should not be worse by design. Question: does it add points on test?
Caveat: synthesis is stochastic, so the two arms carry compile variance.
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
CAP, SAMPLE_PER = 25, 6
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
REG = {
    "hacking": "Intrusion via technical means: stolen/brute-forced credentials, exploited vulnerabilities, SQL injection, remote access.",
    "malware": "Malicious software: ransomware, spyware, trojans, keyloggers, worms installed on a system.",
    "error": "Unintentional mistake: misdelivery, misconfiguration, lost device, accidental publishing or disposal of data.",
    "misuse": "Authorized insider abusing access: snooping, data theft by staff, policy violation.",
    "physical": "Physical action: theft of a device/document, tampering, surveillance, break-in.",
    "social": "Social engineering of a person: phishing, pretexting, bribery, deception.",
}


def _get(url, timeout=90):
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
        sample += [(c, t) for t in rows[:SAMPLE_PER]]
        for t in rows[SAMPLE_PER:]:
            test.append(t); truth.append(c)
    n = len(test)
    print(f"test records: {n}   random: 16.7%   "
          f"majority: {round(collections.Counter(truth).most_common(1)[0][1]/n*100,1)}%\n", flush=True)

    def acc(distill, tag):
        recs = [{"text": t[:1500]} for _, t in sample] + [{"text": t[:1500]} for t in test]
        off = len(sample)
        res = strot.run(recs, "Classify each security-incident report by its primary attack action.",
                        REG, question="What was the primary attack action in this security incident?",
                        measure_labels=True, distill=distill, llm=LLM,
                        state=f"/tmp/cyd_{tag}", sample_size=off)
        routed = {o["index"]: o["action"] for o in res.outcomes}
        correct = sum(1 for i in range(n) if routed.get(i + off) == truth[i])
        fb = sum(1 for f in res.fallbacks if f["index"] >= off)
        return round(correct / n * 100, 1), fb

    a0, fb0 = acc(False, "nodistill")
    print(f"no-distill: {a0}%  (abstain {fb0})", flush=True)
    a1, fb1 = acc(True, "distill")
    print(f"distill:    {a1}%  (abstain {fb1})", flush=True)
    print(f"\ndelta: {round(a1 - a0, 1):+} pp")


main()
