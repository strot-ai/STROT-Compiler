"""Timed LoRA on roberta-base (CPU) -- replace the estimate with a real number.

Same config as the 69.1% proof (seed 11, CAP=90, 4 epochs). Adds explicit
timers: (a) pure LoRA training loop, (b) total wall-clock incl. download/eval.
Prints env + thread count so the CPU number is interpretable.
"""

import collections
import io
import json
import os
import time
import urllib.request
import zipfile

import torch

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
MODEL = "FacebookAI/roberta-base"
CAP = 90
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
SEED = 11
EPOCHS = 4


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
    t_total = time.time()
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"torch threads: {torch.get_num_threads()}  cpu_count: {os.cpu_count()}", flush=True)

    path = snapshot_download(MODEL)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
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

    def enc(b):
        return tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt")

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n, generator=g).tolist()
    tr, te = perm[:int(n*.72)], perm[int(n*.72):]
    yt = [labels[i] for i in te]
    print(f"records {n}  train {len(tr)}  test {len(te)}  epochs {EPOCHS}", flush=True)

    m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=6, local_files_only=True)
    lc = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
                    target_modules=["query", "value"], modules_to_save=["classifier"])
    m = get_peft_model(m, lc)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)", flush=True)

    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    lf = torch.nn.CrossEntropyLoss()

    t_train = time.time()
    m.train()
    for ep in range(EPOCHS):
        order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(SEED*9+ep)).tolist()
        for i in range(0, len(tr), 16):
            bi = [tr[order[k]] for k in range(i, min(i+16, len(tr)))]
            opt.zero_grad()
            lf(m(**enc([texts[j] for j in bi])).logits,
               torch.tensor([labels[j] for j in bi])).backward()
            opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} done @ {time.time()-t_train:.1f}s", flush=True)
    train_secs = time.time() - t_train

    m.eval(); pred = []
    with torch.no_grad():
        for i in range(0, len(te), 16):
            lg = m(**enc([texts[te[j]] for j in range(i, min(i+16, len(te)))])).logits
            pred += lg.argmax(1).tolist()
    acc = sum(1 for j in range(len(te)) if pred[j] == yt[j]) / len(te) * 100

    print(f"\n=== RESULT ===", flush=True)
    print(f"accuracy: {round(acc,1)}%", flush=True)
    print(f"LoRA TRAINING time (pure loop): {train_secs:.1f}s  ({train_secs/60:.2f} min)", flush=True)
    print(f"TOTAL wall-clock (download+train+eval): {time.time()-t_total:.1f}s  ({(time.time()-t_total)/60:.2f} min)", flush=True)


main()
