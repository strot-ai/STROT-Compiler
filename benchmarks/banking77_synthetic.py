"""Banking77 synthetic cold-start: can an LLM-GENERATED training set (zero real
data) train a usable 77-class decision model?

The "synthetic solves zero-shot" hypothesis. For each of the 77 intents the LLM
invents N utterances; we embed them with the frozen encoder, train a head, and
evaluate on the REAL Banking77 test set. Compare synthetic-trained accuracy to
the real-data ceiling (banking77_probe.py) at the same N/class.

No real training labels are ever used -- only the intent NAMES (the label space),
which is exactly what a cold-start user has: they know their labels, not their data.
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
GEN_MODEL = os.environ.get("GEN_MODEL", "claude-haiku-4-5")
PER_CLASS_GEN = 20
PER_CLASS_EVAL = [5, 10, 20]
SEEDS = [11, 23, 37]
MAXLEN = 64
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
client = Anthropic()


def humanize(name):
    return name.replace("_", " ")


def gen_one(name):
    prompt = (
        f"You are generating training data for a banking customer-support intent "
        f"classifier. Intent: \"{humanize(name)}\". Write {PER_CLASS_GEN} diverse, "
        f"realistic one-sentence customer messages that express ONLY this intent. "
        f"Vary phrasing, formality, and detail; no numbering. Return ONLY a JSON "
        f"array of {PER_CLASS_GEN} strings."
    )
    for _ in range(3):
        try:
            msg = client.messages.create(
                model=GEN_MODEL, max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            txt = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "")
            m = re.search(r"\[.*\]", txt, re.S)
            arr = json.loads(m.group(0) if m else txt)
            arr = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
            if arr:
                return name, arr[:PER_CLASS_GEN]
        except Exception as e:
            last = e
    print(f"  gen FAILED {name}: {type(last).__name__}", flush=True)
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


def main():
    ds = load_dataset("PolyAI/banking77")
    names = ds["train"].features["label"].names
    n_cls = len(names)
    te_text, te_y = ds["test"]["text"], ds["test"]["label"]
    print(f"device: {DEVICE}  |  {n_cls} intents  |  generating {PER_CLASS_GEN}/class with {GEN_MODEL}\n", flush=True)

    t0 = time.time()
    syn = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for name, arr in ex.map(gen_one, names):
            syn[name] = arr
    got = sum(len(v) for v in syn.values())
    print(f"generated {got} synthetic utterances across {n_cls} intents in {time.time()-t0:.0f}s\n", flush=True)

    texts, labels = [], []
    for i, name in enumerate(names):
        for s in syn[name]:
            texts.append(s); labels.append(i)
    labels = np.array(labels)

    Xsyn = embed(texts)
    Xte = embed(te_text)

    print("SYNTHETIC-trained head -> evaluated on REAL Banking77 test:")
    for pc in PER_CLASS_EVAL:
        accs = []
        for s in SEEDS:
            rng = np.random.default_rng(s)
            idx = []
            for c in range(n_cls):
                ci = np.where(labels == c)[0]
                idx.extend(rng.choice(ci, size=min(pc, len(ci)), replace=False).tolist())
            head, mu, sd = train_head(Xsyn[idx], labels[idx], n_cls, seed=s)
            accs.append(acc(head, mu, sd, Xte, te_y))
        print(f"  {pc:>2}/class:  {np.mean(accs)*100:5.1f}%   (seeds {[round(a*100,1) for a in accs]})", flush=True)
    print("\n  (run banking77_probe.py for the real-data ceiling on the same encoder)", flush=True)


if __name__ == "__main__":
    main()
