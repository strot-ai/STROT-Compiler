"""Standalone probe for models needing the latest transformers (UltraBERT ships
custom code that the strot-julia image's pinned transformers can't load).

Runs in a fresh python:3.11-slim + latest torch/transformers. Same VCDB data,
same probe, same seeds as cyber_encoders.py, so the numbers drop straight into
that table. SecureBERT2.0 is re-probed here as an ANCHOR -- if it reproduces
~62%, the UltraBERT numbers are comparable across containers.

Reference (from the strot-julia image):
  Julia-1 (general, 144M)        50.5%
  SecureBERT2.0 (security,125M)  61.7%
"""

import collections
import io
import json
import time
import urllib.request
import zipfile

import torch

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
CAP = 45
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
SEEDS = [11, 23, 37, 52, 71]
MODELS = [
    "cisco-ai/SecureBERT2.0-base",   # ANCHOR (should reproduce ~62%)
    "dmis-lab/UltraBERT-150M",       # new general, ~Julia size
    "dmis-lab/UltraBERT-380M",       # new general, big (the size test)
]


def _get(url, timeout=120):
    for a in range(5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == 4:
                raise
            time.sleep(8 * (a + 1))


def hf_embedder(model_id):
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    path = snapshot_download(model_id)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=True).eval()

    def embed(texts):
        vecs = []
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            with torch.no_grad():
                out = mdl(**enc)
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            vecs += pooled.tolist()
        return vecs
    return embed


def probe(X, y, Xte, yte, n_cls=6):
    X, Xte = torch.tensor(X), torch.tensor(Xte)
    y, yte = torch.tensor(y), torch.tensor(yte)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
    X, Xte = (X - mu) / sd, (Xte - mu) / sd
    clf = torch.nn.Linear(X.shape[1], n_cls)
    opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=0.03)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(400):
        opt.zero_grad()
        lossf(clf(X), y).backward()
        opt.step()
    with torch.no_grad():
        return (clf(Xte).argmax(1) == yte).float().mean().item()


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
    texts, labels = [], []
    for c in CLASSES:
        for s in buckets[c]:
            texts.append(s); labels.append(IDX[c])
    n = len(texts)
    print(f"records: {n}   random: 16.7%\n", flush=True)
    print(f"REF  Julia-1 (general,144M) 50.5%   SecureBERT2.0 61.7% (strot-julia image)\n", flush=True)

    for mid in MODELS:
        try:
            embed = hf_embedder(mid)
            vecs = embed(texts)
        except Exception as exc:
            print(f"{mid:34} SKIPPED: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            continue
        accs = []
        for seed in SEEDS:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n, generator=g).tolist()
            cut = int(n * 0.65)
            tr, te = perm[:cut], perm[cut:]
            accs.append(probe([vecs[i] for i in tr], [labels[i] for i in tr],
                              [vecs[i] for i in te], [labels[i] for i in te]))
        print(f"{mid:34} {round(sum(accs)/len(accs)*100,1)}%   "
              f"(seeds: {[round(a*100,1) for a in accs]})", flush=True)


main()
