"""No-fine-tuning study on frozen ModernBERT (fast on CPU -- embed once, train
tiny heads on cached embeddings; the encoder is never backpropped).

1. DIAGNOSTIC: hard contrast pairs. Flip ONLY intent (accidentally<->deliberately)
   and measure how much the frozen embedding moves, vs how far apart random
   different incidents are. If the intent-flip barely moves it, the encoder is
   blind to intent (representation-bound).
2. linear  : logistic regression on frozen embeddings (~the 65% probe)
3. mlp     : nonlinear projection head + cross-entropy
4. supcon  : projection head + supervised-contrastive loss + CE (reviewer's idea)

Reference: frozen linear ~65%, LoRA (fine-tuned) ~70%. Can a frozen projection
head close the gap WITHOUT touching the encoder?
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
MODEL = "answerdotai/ModernBERT-base"
CAP = 50
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
SEEDS = [11, 23, 37]

# hard contrast pairs: (error-version, hacking-version) -- identical but for intent
PAIRS = [
    ("An administrator accidentally disabled authentication, exposing customer records.",
     "An attacker deliberately disabled authentication, exposing customer records."),
    ("An employee mistakenly emailed a spreadsheet of SSNs to the wrong recipient.",
     "An intruder exfiltrated a spreadsheet of SSNs and sent it to an external server."),
    ("A staff member inadvertently left a database open to the public internet.",
     "A hacker exploited an open database to pull records from the public internet."),
    ("An IT worker accidentally deleted the firewall rules during maintenance.",
     "An adversary intentionally deleted the firewall rules to gain access."),
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


def main():
    from transformers import AutoModel, AutoTokenizer
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(MODEL)
    except Exception:
        path = MODEL
    tok = AutoTokenizer.from_pretrained(path)
    enc_model = AutoModel.from_pretrained(path).eval()

    def embed(texts):
        vecs = []
        for i in range(0, len(texts), 16):
            e = tok(texts[i:i+16], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                h = enc_model(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            vecs += ((h*m).sum(1) / m.sum(1).clamp(min=1.0)).tolist()
        return torch.tensor(vecs)

    # ---- DIAGNOSTIC ----
    print("=== DIAGNOSTIC: does the frozen encoder respond to intent? ===", flush=True)
    err = embed([p[0] for p in PAIRS]); hak = embed([p[1] for p in PAIRS])
    errn, hakn = F.normalize(err, dim=1), F.normalize(hak, dim=1)
    intent_delta = (1 - (errn * hakn).sum(1)).mean().item()  # cos distance within intent-pairs
    # baseline: distance between random DIFFERENT pair sentences
    allv = torch.cat([errn, hakn], 0)
    rand = (1 - (allv @ allv.T)).mean().item()  # avg pairwise cos distance
    print(f"  intent-flip cos-distance (accident<->attack): {intent_delta:.3f}")
    print(f"  avg cos-distance between arbitrary sentences:  {rand:.3f}")
    print(f"  ratio (intent-flip / arbitrary): {intent_delta/rand:.2f}  "
          f"({'encoder BARELY responds to intent' if intent_delta/rand < 0.5 else 'encoder responds to intent'})\n", flush=True)

    # ---- data ----
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
    n = len(texts)
    X = embed(texts)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
    X = (X - mu) / sd
    y = torch.tensor(labels)
    print(f"records: {n}  (frozen ModernBERT embeddings, encoder never trained)\n", flush=True)

    def supcon(z, yy, temp=0.1):
        z = F.normalize(z, dim=1)
        sim = z @ z.T / temp
        N = z.shape[0]
        self_mask = torch.eye(N).bool()
        sim.masked_fill_(self_mask, -1e9)
        pos = (yy.view(-1, 1) == yy.view(1, -1)) & ~self_mask
        logprob = sim - torch.logsumexp(sim, 1, keepdim=True)
        return -((logprob * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()

    def run_arm(kind, Xtr, ytr, Xte, yte, d):
        if kind == "linear":
            net = torch.nn.Linear(d, 6)
            params = net.parameters()
            def forward(x): return net(x)
        else:
            back = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU())
            proj = torch.nn.Linear(256, 128)
            clf = torch.nn.Linear(256, 6)
            params = list(back.parameters()) + list(proj.parameters()) + list(clf.parameters())
            def forward(x): return clf(back(x))
            def project(x): return proj(back(x))
        opt = torch.optim.Adam(params, lr=0.005, weight_decay=0.02)
        ce = torch.nn.CrossEntropyLoss()
        for _ in range(500):
            opt.zero_grad()
            loss = ce(forward(Xtr), ytr)
            if kind == "supcon":
                loss = loss + 0.5 * supcon(project(Xtr), ytr)
            loss.backward(); opt.step()
        with torch.no_grad():
            return (forward(Xte).argmax(1) == yte).float().mean().item() * 100

    d = X.shape[1]
    for kind in ("linear", "mlp", "supcon"):
        accs = []
        for seed in SEEDS:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n, generator=g).tolist()
            cut = int(n*.65); tr, te = perm[:cut], perm[cut:]
            accs.append(run_arm(kind, X[tr], y[tr], X[te], y[te], d))
        print(f"{kind:7} {round(sum(accs)/len(accs),1)}%   (seeds {[round(a,1) for a in accs]})", flush=True)


main()
