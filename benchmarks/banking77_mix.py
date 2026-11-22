"""Banking77 synthetic + few-real MIX: the onboarding curve.

Cold start is synthetic-only (~70%). The product question: as a user labels a
FEW real examples per class, how fast does accuracy climb -- and does the
synthetic base HELP at each step, or does real data just dominate (or does
synthetic DRAG it down, as in the original fine-tune finding)?

For k in {0,2,5,10} real/class:
    real_only(k)        -- head on k real examples/class
    synth(20)+real(k)   -- head on 20 synthetic/class + k real/class
Evaluated on the REAL Banking77 test set.
"""
import concurrent.futures as cf
import json
import os
import re
import time

import numpy as np
import torch
from anthropic import Anthropic
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

ENCODER = "Alibaba-NLP/gte-modernbert-base"
GEN_MODEL = os.environ.get("GEN_MODEL", "claude-opus-4-8")
GEN_N = 20
K_REAL = [0, 2, 5, 10]
SEEDS = [11, 23, 37]
MAXLEN = 64
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
client = Anthropic()


def gen_one(name):
    prompt = (
        f"You are generating training data for a banking customer-support intent "
        f"classifier. Intent: \"{name.replace('_', ' ')}\". Write {GEN_N} diverse, "
        f"realistic one-sentence customer messages that express ONLY this intent. "
        f"Vary phrasing, formality, and detail; no numbering. Return ONLY a JSON "
        f"array of {GEN_N} strings."
    )
    for _ in range(3):
        try:
            msg = client.messages.create(model=GEN_MODEL, max_tokens=1500,
                                         messages=[{"role": "user", "content": prompt}])
            txt = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "")
            m = re.search(r"\[.*\]", txt, re.S)
            arr = [s.strip() for s in json.loads(m.group(0) if m else txt) if isinstance(s, str) and s.strip()]
            if arr:
                return name, arr[:GEN_N]
        except Exception:
            pass
    return name, []


def embed(texts, bs=64):
    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER, trust_remote_code=True).to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True,
                      max_length=MAXLEN, return_tensors="pt").to(DEVICE)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu().float().numpy())
    del model
    return np.concatenate(out, 0)


def train_head(X, y, n_cls, seed=0, epochs=300, lr=0.05, wd=1e-3):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp(min=1e-6)
    Xt = (Xt - mu) / sd
    head = torch.nn.Linear(Xt.shape[1], n_cls)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    lf = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); lf(head(Xt), yt).backward(); opt.step()
    return head, mu, sd


def acc(head, mu, sd, X, y):
    Xt = (torch.tensor(X, dtype=torch.float32) - mu) / sd
    with torch.no_grad():
        pred = head(Xt).argmax(1).numpy()
    return float((pred == np.array(y)).mean())


def sample_k(y, k, n_cls, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(n_cls):
        ci = np.where(y == c)[0]
        if k > 0 and len(ci):
            idx.extend(rng.choice(ci, size=min(k, len(ci)), replace=False).tolist())
    return np.array(idx, dtype=int)


def main():
    ds = load_dataset("PolyAI/banking77")
    names = ds["train"].features["label"].names
    n_cls = len(names)
    tr_y = np.array(ds["train"]["label"])
    te_text, te_y = ds["test"]["text"], ds["test"]["label"]
    print(f"device: {DEVICE} | teacher: {GEN_MODEL} | {n_cls} intents\n", flush=True)

    t0 = time.time()
    syn = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for name, arr in ex.map(gen_one, names):
            syn[name] = arr
    syn_texts, syn_lab = [], []
    for i, name in enumerate(names):
        for s in syn[name]:
            syn_texts.append(s); syn_lab.append(i)
    syn_lab = np.array(syn_lab)
    print(f"generated {len(syn_texts)} synthetic in {time.time()-t0:.0f}s\n", flush=True)

    Xsyn = embed(syn_texts)
    Xtr = embed(ds["train"]["text"])
    Xte = embed(te_text)

    print(f"{'k real/class':>13} | {'real_only':>10} | {'synth+real':>11} | delta")
    print("-" * 52)
    for k in K_REAL:
        ro, sr = [], []
        for s in SEEDS:
            ridx = sample_k(tr_y, k, n_cls, s)
            # real_only (k=0 is undefined -> skip, show as --)
            if k > 0:
                h, mu, sd = train_head(Xtr[ridx], tr_y[ridx], n_cls, seed=s)
                ro.append(acc(h, mu, sd, Xte, te_y))
            # synth + real
            if k > 0:
                X = np.concatenate([Xsyn, Xtr[ridx]]); Y = np.concatenate([syn_lab, tr_y[ridx]])
            else:
                X, Y = Xsyn, syn_lab
            h, mu, sd = train_head(X, Y, n_cls, seed=s)
            sr.append(acc(h, mu, sd, Xte, te_y))
        ro_m = f"{np.mean(ro)*100:5.1f}%" if ro else "   -- "
        d = f"{(np.mean(sr)-np.mean(ro))*100:+5.1f}" if ro else "  -- "
        print(f"{k:>13} | {ro_m:>10} | {np.mean(sr)*100:9.1f}%  | {d}", flush=True)
    print("\n(delta > 0: synthetic base helps at this real-data level; ~0: neutral; < 0: it drags)", flush=True)


if __name__ == "__main__":
    main()
