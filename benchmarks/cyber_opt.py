"""Combined cyber optimization probe: (a) richer projection + (b) fair distill.

Same real VCDB held-out test set, real labels as the ONLY judge. Four arms:

  flat          summary only, no distill            (control, ~40-46%)
  distill       summary only, distill=True          (b) -- now with MORE teacher
                                                     data (does scarcity explain
                                                     the earlier -10.6pp?)
  rich          summary + non-label fields, no dist  (a) -- richer input to the
                                                     frozen encoder
  rich+distill  both                                 synergy

(b): teacher set raised to ~20/class so the distilled head + its gate fold have
enough real data to be reliable (the 36-record run overfit).
(a): richer text prepends readable strings from asset + discovery_method. These
are NOT the action label (action.* is excluded), but they mildly correlate --
legitimate signal a production system would use; reported honestly as such.
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
CAP, SAMPLE_PER = 45, 20
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


def _get(url, timeout=90):
    for a in range(5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == 4:
                raise
            time.sleep(8 * (a + 1))


def leaf_strings(node, out=None):
    out = [] if out is None else out
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for v in node.values():
            leaf_strings(v, out)
    elif isinstance(node, list):
        for v in node:
            leaf_strings(v, out)
    return out


def rich_text(r):
    s = (r.get("summary") or "").strip()
    asset = ", ".join(dict.fromkeys(leaf_strings(r.get("asset") or {})))[:200]
    disc = ", ".join(dict.fromkeys(leaf_strings(r.get("discovery_method") or {})))[:120]
    extra = ""
    if asset:
        extra += f" | assets affected: {asset}"
    if disc:
        extra += f" | discovered via: {disc}"
    return s + extra


def main():
    data = json.loads(zipfile.ZipFile(io.BytesIO(_get(VCDB_ZIP))).read("vcdb_1-of-1.json"))

    def label(r):
        acts = [a for a in r.get("action", {}) if a in CLASSES]
        return acts[0] if len(acts) == 1 else None

    # keep the raw record so we can build both plain and rich text from the SAME rows
    buckets = collections.defaultdict(list)
    for r in data:
        lab = label(r)
        s = r.get("summary")
        if lab in CLASSES and isinstance(s, str) and len(s.strip()) > 40 and len(buckets[lab]) < CAP:
            buckets[lab].append(r)

    sample_recs, test_recs, truth = [], [], []
    for c in CLASSES:
        rows = buckets[c]
        sample_recs += rows[:SAMPLE_PER]
        for r in rows[SAMPLE_PER:]:
            test_recs.append(r); truth.append(c)
    n = len(test_recs)
    counts = {c: len(buckets[c]) for c in CLASSES}
    print(f"classes: {counts}")
    print(f"teacher sample: {len(sample_recs)}   test: {n}   random: 16.7%   "
          f"majority: {round(collections.Counter(truth).most_common(1)[0][1]/n*100,1)}%\n", flush=True)

    def run_arm(text_fn, distill, tag):
        recs = [{"text": text_fn(r)[:1800]} for r in sample_recs] + \
               [{"text": text_fn(r)[:1800]} for r in test_recs]
        off = len(sample_recs)
        res = strot.run(recs, GOAL, REG, question=Q, measure_labels=True,
                        distill=distill, llm=LLM, state=f"/tmp/co_{tag}", sample_size=off)
        routed = {o["index"]: o["action"] for o in res.outcomes}
        correct = sum(1 for i in range(n) if routed.get(i + off) == truth[i])
        return round(correct / n * 100, 1)

    plain = lambda r: (r.get("summary") or "").strip()
    a_flat = run_arm(plain, False, "flat")
    print(f"flat         (summary, no distill):   {a_flat}%", flush=True)
    a_dist = run_arm(plain, True, "distill")
    print(f"distill      (summary, distill):       {a_dist}%  ({a_dist-a_flat:+} vs flat)", flush=True)
    a_rich = run_arm(rich_text, False, "rich")
    print(f"rich         (+fields, no distill):    {a_rich}%  ({a_rich-a_flat:+} vs flat)", flush=True)
    a_both = run_arm(rich_text, True, "both")
    print(f"rich+distill (+fields, distill):       {a_both}%  ({a_both-a_flat:+} vs flat)", flush=True)


main()
