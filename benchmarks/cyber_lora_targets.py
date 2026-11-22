"""LoRA target-module study on ModernBERT: does targeting the FFN (MLP) beat
targeting attention for cyber? Also confirms ModernBERT fine-tunes cleanly w/ peft.

Arms (same data/splits, real labels, Path-3 gate):
  frozen   : head only (reference)
  attn     : LoRA on attention projections only
  ffn      : LoRA on FFN/MLP projections only   <- the ask (no attention)
  both     : LoRA on attention + FFN

Target modules are detected at runtime from the actual model (ModernBERT naming
differs from roberta; ".attn." vs ".mlp." disambiguates, incl. the shared 'Wo').
"""

import collections
import io
import json
import time
import urllib.request
import zipfile

import torch
import torch.nn as nn

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
MODEL = "answerdotai/ModernBERT-base"
CAP = 90
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
SEEDS = [11]
ALPHAS = [0.10, 0.05]
EPOCHS = 3


def _get(url, timeout=120):
    for a in range(5):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == 4:
                raise
            time.sleep(8 * (a + 1))


def find_cutoff(conf, pred, ytrue, n_total, alpha):
    order = sorted(range(len(conf)), key=lambda i: conf[i], reverse=True)
    wrong, cutoff = 0, 2.0
    for i in order:
        if pred[i] != ytrue[i]:
            wrong += 1
        if wrong / n_total <= alpha:
            cutoff = conf[i]
        else:
            break
    return cutoff


def main():
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
            buckets[lab].append(s.strip()[:1500])
    texts, labels = [], []
    for c in CLASSES:
        for s in buckets[c]:
            texts.append(s); labels.append(IDX[c])
    n = len(texts)
    print(f"records: {n}\n", flush=True)

    # detect target modules from the actual model
    probe = AutoModelForSequenceClassification.from_pretrained(path, num_labels=6, local_files_only=True)
    lin = [nm for nm, m in probe.named_modules() if isinstance(m, nn.Linear) and ("layers." in nm or "encoder." in nm)]
    attn_mods = sorted({nm for nm in lin if ".attn" in nm})
    ffn_mods = sorted({nm for nm in lin if ".mlp" in nm})
    print("layer-0 Linear modules:", [nm for nm in lin if ".0." in nm])
    print(f"attn target count: {len(attn_mods)}   ffn target count: {len(ffn_mods)}\n", flush=True)
    del probe

    ARMS = {
        "attn": attn_mods,
        "ffn": ffn_mods,
        "both": sorted(set(attn_mods) | set(ffn_mods)),
    }

    def enc(batch):
        return tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")

    def conf_pred(model, idxs, T=1.0):
        model.eval(); conf, pred = [], []
        with torch.no_grad():
            for i in range(0, len(idxs), 16):
                lg = model(**enc([texts[j] for j in idxs[i:i+16]])).logits / T
                c, p = torch.softmax(lg, 1).max(1)
                conf += c.tolist(); pred += p.tolist()
        return conf, pred

    def train(model, tr, seed):
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        lf = torch.nn.CrossEntropyLoss(); model.train()
        for ep in range(EPOCHS):
            order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed*9+ep)).tolist()
            for i in range(0, len(tr), 16):
                bi = [tr[order[k]] for k in range(i, min(i+16, len(tr)))]
                opt.zero_grad()
                lf(model(**enc([texts[j] for j in bi])).logits,
                   torch.tensor([labels[j] for j in bi])).backward()
                opt.step()

    def gate(tag, c_cal, p_cal, y_cal, c_te, p_te, y_te):
        nt = len(y_te)
        acc = sum(1 for j in range(nt) if p_te[j] == y_te[j]) / nt * 100
        out = f"  {tag:6} top1 {round(acc,1)}%"
        for a in ALPHAS:
            cut = find_cutoff(c_cal, p_cal, y_cal, len(y_cal), a)
            disp = [j for j in range(nt) if c_te[j] >= cut]
            corr = sum(1 for j in disp if p_te[j] == y_te[j])
            out += f"  |  a={a}: {round(len(disp)/nt*100,1)}% @ {round(corr/len(disp)*100,1) if disp else 0}%"
        print(out, flush=True)

    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        tr, cal, te = perm[:int(n*.55)], perm[int(n*.55):int(n*.72)], perm[int(n*.72):]
        y_cal, y_te = [labels[i] for i in cal], [labels[i] for i in te]
        print(f"--- seed {seed} (train {len(tr)}/cal {len(cal)}/test {len(te)}) ---", flush=True)

        # frozen head-only
        m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=6, local_files_only=True)
        for p in m.base_model.parameters():
            p.requires_grad = False
        train(m, tr, seed)
        cc, pc = conf_pred(m, cal); ct, pt = conf_pred(m, te)
        gate("frozen", cc, pc, y_cal, ct, pt, y_te)

        for tag, targets in ARMS.items():
            m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=6, local_files_only=True)
            lc = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
                            target_modules=targets, modules_to_save=["classifier"])
            m = get_peft_model(m, lc)
            train(m, tr, seed)
            cc, pc = conf_pred(m, cal); ct, pt = conf_pred(m, te)
            gate(tag, cc, pc, y_cal, ct, pt, y_te)
        print(flush=True)


main()
