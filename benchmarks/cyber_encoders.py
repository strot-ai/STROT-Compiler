"""Linear probe: whose EYES see cyber better -- Julia (general) vs security
domain encoders. The gold-standard test of representation quality.

For each encoder: freeze it, embed the incident text, train a simple linear
classifier on the TRUE labels (train split), evaluate on the TRUE labels (test
split). This measures the CEILING of what each representation can separate --
if a domain encoder's fingerprints pull the cyber classes apart better than
Julia's, swapping it into the nlu= slot would help.

No LLM: this isolates representation quality. Averaged over 3 seeds (small folds
are noisy -- the lesson from the distill run). Same balanced VCDB data for all.
"""

import collections
import io
import json
import time
import urllib.request
import zipfile

import torch
from strot.distill import make_encoder_embedder

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
CAP = 45
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
SEEDS = [11, 23, 37, 52, 71]
DOMAIN = [
    "FacebookAI/roberta-base",          # general, 125M -- CONTROL for SecureBERT2.0 (same arch/size)
    "FacebookAI/roberta-large",         # general, 355M -- size test (big general)
    "ehsanaghaei/SecureBERT",           # security domain
    "cisco-ai/SecureBERT2.0-base",      # security domain (a security-pretrained roberta-base)
    "markusbayer/CySecBERT",            # security domain
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
    # Download files only, then load offline -- avoids the hub metadata call
    # (discussions endpoint) that 403s and crashes on some repos.
    path = snapshot_download(model_id, allow_patterns=["*.json", "*.txt", "*.model", "*.py",
                                                       "*.safetensors", "*.bin", "vocab*", "merges*", "tokenizer*"])
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=True).eval()

    def embed(texts):
        vecs = []
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], padding=True, truncation=True,
                      max_length=256, return_tensors="pt")
            with torch.no_grad():
                h = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            vecs += pooled.tolist()
        return vecs
    return embed


def probe(X, y, Xte, yte, n_cls=6):
    """Standardized, L2-regularized linear classifier. Returns test accuracy."""
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
    print(f"records: {n}  per-class: {{c: len}} {{{', '.join(f'{c}:{len(buckets[c])}' for c in CLASSES)}}}")
    print(f"random: 16.7%   (linear probe = best a frozen encoder can do with true labels)\n", flush=True)

    embedders = [("Julia-1 (general)", lambda: make_encoder_embedder("/opt/julia-1"))]
    for m in DOMAIN:
        embedders.append((m, (lambda mm: (lambda: hf_embedder(mm)))(m)))

    for name, make in embedders:
        try:
            embed = make()
            vecs = embed(texts)  # embed once, reuse across seeds
        except Exception as exc:
            print(f"{name:34} SKIPPED: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            continue
        accs = []
        for seed in SEEDS:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n, generator=g).tolist()
            cut = int(n * 0.65)
            tr, te = perm[:cut], perm[cut:]
            acc = probe([vecs[i] for i in tr], [labels[i] for i in tr],
                        [vecs[i] for i in te], [labels[i] for i in te])
            accs.append(acc)
        mean = sum(accs) / len(accs)
        print(f"{name:34} {round(mean*100,1)}%   (seeds: {[round(a*100,1) for a in accs]})", flush=True)


main()
