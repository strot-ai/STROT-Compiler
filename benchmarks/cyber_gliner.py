"""GLiNER-as-preprocessing test on cyber: does extracting structured entities
first help classification, vs raw summary?

Three arms, same encoder (roberta-base frozen, mean-pool) + linear probe, true
labels, 3 seeds:
  raw     : raw summary text
  gliner  : GLiNER-extracted entity features only
  hybrid  : raw summary + GLiNER features

NEUTRAL extraction schema (generic entities, NOT the attack categories -- using
the category names would leak the label). Honest prediction: ~neutral on these
prose summaries (the bottleneck is intent, not hidden entities), but GLiNER would
help on entity-rich data (raw logs, adjuster notes). Measuring to confirm.
"""

import collections
import io
import json
import time
import urllib.request
import zipfile

import torch

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
ENC = "FacebookAI/roberta-base"
GLINER = "urchade/gliner_medium-v2.1"
CAP = 45
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
SEEDS = [11, 23, 37]
# neutral entities -- descriptive, not the attack categories (no leakage)
NER_LABELS = ["person or actor", "organization or company", "device or system",
              "document or data", "software or program", "physical location",
              "method or technique", "monetary amount"]


def _get(url, timeout=120):
    for a in range(5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == 4:
                raise
            time.sleep(8 * (a + 1))


def roberta_embed(path, texts):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    mdl = AutoModel.from_pretrained(path, local_files_only=True).eval()
    vecs = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i+16], padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        vecs += ((h*m).sum(1) / m.sum(1).clamp(min=1.0)).tolist()
    return vecs


def probe(vecs, labels, n):
    accs = []
    X = torch.tensor(vecs)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
    X = (X - mu) / sd
    y = torch.tensor(labels)
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        cut = int(n*.65); tr, te = perm[:cut], perm[cut:]
        clf = torch.nn.Linear(X.shape[1], 6)
        opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=0.03)
        lf = torch.nn.CrossEntropyLoss()
        Xtr, ytr = X[tr], y[tr]
        for _ in range(400):
            opt.zero_grad(); lf(clf(Xtr), ytr).backward(); opt.step()
        with torch.no_grad():
            accs.append((clf(X[te]).argmax(1) == y[te]).float().mean().item()*100)
    return sum(accs)/len(accs), [round(a,1) for a in accs]


def main():
    from huggingface_hub import snapshot_download
    from gliner import GLiNER

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
    raw, labels = [], []
    for c in CLASSES:
        for s in buckets[c]:
            raw.append(s); labels.append(IDX[c])
    n = len(raw)
    print(f"records: {n}\n", flush=True)

    # GLiNER extraction
    gl = GLiNER.from_pretrained(GLINER)
    feats = []
    for t in raw:
        ents = gl.predict_entities(t, NER_LABELS, threshold=0.4)
        feats.append("; ".join(f"{e['label']}: {e['text']}" for e in ents) or "none")
    hybrid = [r + " || " + f for r, f in zip(raw, feats)]
    # peek at a couple extractions
    print("sample extractions:")
    for i in (0, n//2):
        print(f"  [{CLASSES[labels[i]]}] {raw[i][:80]}")
        print(f"     -> {feats[i][:120]}")
    print(flush=True)

    path = snapshot_download(ENC)
    for tag, texts in [("raw", raw), ("gliner", feats), ("hybrid", hybrid)]:
        acc, seeds = probe(roberta_embed(path, texts), labels, n)
        print(f"{tag:8} {round(acc,1)}%   (seeds {seeds})", flush=True)


main()
