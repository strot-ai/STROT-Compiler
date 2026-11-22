"""No-fine-tuning, NON-embedding study on cyber: cross-encoder / NLI reasoning.

The frozen bi-encoder work (linear 57.5 / mlp 58.4 / supcon 57.1) pools the
incident into ONE vector independently of the label, then asks "are these two
vectors close?". This instead reads incident + hypothesis JOINTLY and asks
"does this evidence support this decision?" -- a different mechanism entirely.
No training here: pretrained checkpoints (MNLI / ranking), scored zero-shot.

Arms:
  nli   : roberta-large-mnli. Score P(entailment) of a per-class hypothesis
          against the incident (premise). argmax over 6 classes.
  rank  : cross-encoder/ms-marco-MiniLM-L-6-v2. Score relevance of incident
          (query) vs a rich per-class description (doc). argmax over 6.

Same 300-record VERIS/VCDB set, real labels. Zero-shot => no fit => eval on all
300. Reports accuracy, macro-F1, per-class recall, and CPU latency/record.
Reference: frozen bi-encoder ~58%, LoRA (task-specific FT) ~70%.
"""

import collections
import io
import json
import time
import urllib.request
import zipfile

import torch
import torch.nn.functional as F

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
NLI_MODEL = "FacebookAI/roberta-large-mnli"
RANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CAP = 50
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}

# per-class hypotheses (NLI) -- decision-critical meaning: intent, actor, mechanism
HYP = {
    "hacking":  "A malicious actor deliberately bypassed or broke security controls to gain unauthorized access.",
    "malware":  "Malicious software such as a virus, ransomware, trojan, or backdoor was used to compromise systems.",
    "error":    "An authorized person made an unintentional mistake or misconfiguration that caused the incident.",
    "misuse":   "An authorized insider abused their legitimate access or privileges for an unauthorized purpose.",
    "physical": "The incident involved physical action such as theft, tampering, or loss of a device or document.",
    "social":   "A person was deceived or manipulated through phishing, pretexting, or other social engineering.",
}
# richer docs for the reranker (query=incident, doc=class description)
DOC = {c: f"{c}: {HYP[c]}" for c in CLASSES}


def _get(url, timeout=120):
    for a in range(5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == 4:
                raise
            time.sleep(8 * (a + 1))


def load_data():
    data = json.loads(zipfile.ZipFile(io.BytesIO(_get(VCDB_ZIP))).read("vcdb_1-of-1.json"))

    def label(r):
        acts = [a for a in r.get("action", {}) if a in CLASSES]
        return acts[0] if len(acts) == 1 else None

    buckets = collections.defaultdict(list)
    for r in data:
        lab = label(r)
        s = r.get("summary")
        if lab in CLASSES and isinstance(s, str) and len(s.strip()) > 40 and len(buckets[lab]) < CAP:
            buckets[lab].append(s.strip()[:1000])
    texts, labels = [], []
    for c in CLASSES:
        for s in buckets[c]:
            texts.append(s); labels.append(IDX[c])
    return texts, labels


def metrics(tag, preds, labels, secs):
    n = len(labels)
    acc = sum(1 for i in range(n) if preds[i] == labels[i]) / n * 100
    f1s = []
    for c in range(6):
        tp = sum(1 for i in range(n) if preds[i] == c and labels[i] == c)
        fp = sum(1 for i in range(n) if preds[i] == c and labels[i] != c)
        fn = sum(1 for i in range(n) if preds[i] != c and labels[i] == c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    macro = sum(f1s) / 6 * 100
    recs = {CLASSES[c]: round(sum(1 for i in range(n) if preds[i] == c and labels[i] == c) /
            max(1, sum(1 for i in range(n) if labels[i] == c)) * 100) for c in range(6)}
    print(f"{tag:5} acc {round(acc,1)}%  macroF1 {round(macro,1)}%  "
          f"{round(secs/n*1000)}ms/rec", flush=True)
    print(f"      per-class recall: {recs}", flush=True)


def main():
    from huggingface_hub import snapshot_download
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

    texts, labels = load_data()
    n = len(texts)
    print(f"records: {n}  (zero-shot, no training)\n", flush=True)

    # ---- NLI arm ----
    npath = snapshot_download(NLI_MODEL)
    ntok = AutoTokenizer.from_pretrained(npath, local_files_only=True)
    nmod = AutoModelForSequenceClassification.from_pretrained(npath, local_files_only=True).eval()
    # roberta-large-mnli label order: 0=contradiction,1=neutral,2=entailment
    ent_idx = 2
    t0 = time.time()
    preds = []
    for t in texts:
        scores = []
        prem = [t] * 6
        hyp = [HYP[c] for c in CLASSES]
        e = ntok(prem, hyp, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logit = nmod(**e).logits
        ent = F.softmax(logit, dim=1)[:, ent_idx]
        preds.append(int(ent.argmax().item()))
    metrics("nli", preds, labels, time.time() - t0)
    print(flush=True)
    del nmod

    # ---- reranker arm ----
    rpath = snapshot_download(RANK_MODEL)
    rtok = AutoTokenizer.from_pretrained(rpath, local_files_only=True)
    rmod = AutoModelForSequenceClassification.from_pretrained(rpath, local_files_only=True).eval()
    t0 = time.time()
    preds = []
    for t in texts:
        q = [t] * 6
        d = [DOC[c] for c in CLASSES]
        e = rtok(q, d, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logit = rmod(**e).logits.squeeze(-1)
        preds.append(int(logit.argmax().item()))
    metrics("rank", preds, labels, time.time() - t0)


main()
