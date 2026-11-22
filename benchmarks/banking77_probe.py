"""Banking77 linear probe: can a HEAD on a frozen encoder handle 77 classes?

The question behind the STROT "specialized decision mini-model" direction:
  - Does a cheap head on a strong frozen encoder clear a HIGH-CARDINALITY task
    (77 intents -- well past Julia's 2-20 option limit)?
  - Which encoder gives the better floor: gte-modernbert (embedding-pretrained)
    vs plain ModernBERT-base (general MLM)?
  - How data-hungry is the head? (N examples/class sweep -> the low-data story.)

No LLM, no fine-tuning. Freeze encoder -> mean-pool -> train a linear head on the
TRUE train labels -> eval on the TRUE test labels. This is the CEILING of what
each frozen representation can separate with a head. Mirrors cyber_encoders.py,
scaled to 77 classes.
"""
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

ENCODERS = [
    ("gte-modernbert-base", "Alibaba-NLP/gte-modernbert-base"),
    ("ModernBERT-base", "answerdotai/ModernBERT-base"),
]
PER_CLASS = [5, 10, 20, None]  # None = all available train examples
SEEDS = [11, 23, 37]
MAXLEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def embed(model_id, texts, bs=64):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True,
                      max_length=MAXLEN, return_tensors="pt").to(DEVICE)
            hidden = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
            out.append(pooled.cpu().float().numpy())
    del model
    return np.concatenate(out, 0)


def train_head(X, y, n_cls, epochs=300, lr=0.05, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    # standardize
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp(min=1e-6)
    Xt = (Xt - mu) / sd
    head = torch.nn.Linear(Xt.shape[1], n_cls)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        lossf(head(Xt), yt).backward()
        opt.step()
    return head, mu, sd


def acc(head, mu, sd, X, y):
    Xt = (torch.tensor(X, dtype=torch.float32) - mu) / sd
    with torch.no_grad():
        pred = head(Xt).argmax(1).numpy()
    return float((pred == np.array(y)).mean())


def subsample(y, per_class, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(np.array(y) == c)[0]
        take = len(ci) if per_class is None else min(per_class, len(ci))
        idx.extend(rng.choice(ci, size=take, replace=False).tolist())
    rng.shuffle(idx)
    return np.array(idx)


def main():
    print(f"device: {DEVICE}\n", flush=True)
    ds = load_dataset("PolyAI/banking77")
    tr_text = ds["train"]["text"]; tr_y = ds["train"]["label"]
    te_text = ds["test"]["text"]; te_y = ds["test"]["label"]
    n_cls = len(set(tr_y))
    maj = max(np.bincount(te_y)) / len(te_y)
    print(f"Banking77: {len(tr_text)} train / {len(te_text)} test, {n_cls} classes")
    print(f"random: {100/n_cls:.1f}%   majority: {maj*100:.1f}%\n", flush=True)

    for name, model_id in ENCODERS:
        t0 = time.time()
        try:
            Xtr = embed(model_id, tr_text)
            Xte = embed(model_id, te_text)
        except Exception as e:
            print(f"{name:22} SKIPPED: {type(e).__name__}: {str(e)[:90]}", flush=True)
            continue
        print(f"{name}  (embed {time.time()-t0:.0f}s, dim {Xtr.shape[1]})", flush=True)
        for pc in PER_CLASS:
            accs = []
            for s in SEEDS:
                idx = subsample(tr_y, pc, s)
                head, mu, sd = train_head(Xtr[idx], np.array(tr_y)[idx], n_cls, seed=s)
                accs.append(acc(head, mu, sd, Xte, te_y))
            tag = "all" if pc is None else f"{pc}/class"
            n_used = len(subsample(tr_y, pc, SEEDS[0]))
            print(f"    {tag:>9} (n={n_used:>5}):  {np.mean(accs)*100:5.1f}%   "
                  f"(seeds {[round(a*100,1) for a in accs]})", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
