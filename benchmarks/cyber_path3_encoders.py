"""Path 3 on BETTER base models: does a stronger frozen encoder rescue the
conformal triage gate on cyber (vs Julia's 0% answered)?

For each encoder: embed -> train a linear head (true labels) -> temperature-scale
-> find the confidence cutoff on a calibration split that bounds wrong-dispatch
<= alpha -> apply to a held-out TEST split and measure:

  answered%  = dispatched locally ($0)        acc@answered = accuracy there
  wrong/ALL  = wrong-dispatched / all test     (must be <= alpha: guarantee)

Best-case: head trained on TRUE labels, so the answered slice is an UPPER bound
(real STROT uses LLM labels, slightly worse). Averaged over seeds. Reference:
Julia answered 0% at both alphas.
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
SEEDS = [11, 23, 37]
ALPHAS = [0.10, 0.05]
MODELS = ["dmis-lab/UltraBERT-380M", "dmis-lab/UltraBERT-150M", "FacebookAI/roberta-large"]


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
            enc = tok(texts[i:i + 16], padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                out = mdl(**enc)
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vecs += ((h * mask).sum(1) / mask.sum(1).clamp(min=1.0)).tolist()
        return vecs
    return embed


def train_head(X, y, n_cls):
    X, y = torch.tensor(X), torch.tensor(y)
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
    Xn = (X - mu) / sd
    clf = torch.nn.Linear(Xn.shape[1], n_cls)
    opt = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=0.03)
    lf = torch.nn.CrossEntropyLoss()
    for _ in range(400):
        opt.zero_grad(); lf(clf(Xn), y).backward(); opt.step()
    return clf, mu, sd


def logits_of(clf, mu, sd, X):
    Xn = (torch.tensor(X) - mu) / sd
    with torch.no_grad():
        return clf(Xn)


def fit_temperature(logits, y):
    y = torch.tensor(y)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([logT], lr=0.05)
    lf = torch.nn.CrossEntropyLoss()
    for _ in range(200):
        opt.zero_grad(); lf(logits / logT.exp(), y).backward(); opt.step()
    return logT.exp().item()


def conf_pred(logits, T):
    p = torch.softmax(logits / T, dim=1)
    conf, pred = p.max(1)
    return conf.tolist(), pred.tolist()


def find_cutoff(conf, pred, ytrue, n_total, alpha):
    """Lowest confidence cutoff s.t. (dispatched & wrong)/n_total <= alpha; maximizes throughput."""
    order = sorted(range(len(conf)), key=lambda i: conf[i], reverse=True)
    wrong = 0
    cutoff = 2.0  # dispatch nothing by default
    for k, i in enumerate(order, start=1):
        if pred[i] != ytrue[i]:
            wrong += 1
        if wrong / n_total <= alpha:
            cutoff = conf[i]  # still within budget at this (lowest-so-far) confidence
        else:
            break
    return cutoff


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
    print(f"records: {n}   (Julia reference: answered 0% at both alphas)\n", flush=True)

    for mid in MODELS:
        try:
            vecs = hf_embedder(mid)(texts)
        except Exception as exc:
            print(f"{mid}: SKIPPED {type(exc).__name__}: {str(exc)[:80]}\n", flush=True)
            continue
        agg = {a: {"ans": [], "acc": [], "wrong": []} for a in ALPHAS}
        for seed in SEEDS:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(n, generator=g).tolist()
            tr, cal, te = perm[:int(n*.4)], perm[int(n*.4):int(n*.7)], perm[int(n*.7):]
            clf, mu, sd = train_head([vecs[i] for i in tr], [labels[i] for i in tr], 6)
            T = fit_temperature(logits_of(clf, mu, sd, [vecs[i] for i in cal]), [labels[i] for i in cal])
            c_cal, p_cal = conf_pred(logits_of(clf, mu, sd, [vecs[i] for i in cal]), T)
            c_te, p_te = conf_pred(logits_of(clf, mu, sd, [vecs[i] for i in te]), T)
            y_cal = [labels[i] for i in cal]
            y_te = [labels[i] for i in te]
            nt = len(te)
            for a in ALPHAS:
                cut = find_cutoff(c_cal, p_cal, y_cal, len(cal), a)
                disp = [j for j in range(nt) if c_te[j] >= cut]
                corr = sum(1 for j in disp if p_te[j] == y_te[j])
                agg[a]["ans"].append(len(disp) / nt * 100)
                agg[a]["acc"].append(corr / len(disp) * 100 if disp else 0.0)
                agg[a]["wrong"].append((len(disp) - corr) / nt * 100)
        print(f"=== {mid} ===", flush=True)
        for a in ALPHAS:
            ans = sum(agg[a]["ans"]) / len(SEEDS)
            acc = sum(agg[a]["acc"]) / len(SEEDS)
            wr = sum(agg[a]["wrong"]) / len(SEEDS)
            print(f"  alpha={a}: answered {round(ans,1)}%   acc@answered {round(acc,1)}%   "
                  f"wrong/all {round(wr,1)}% (<= {int(a*100)}%)", flush=True)
        print(flush=True)


main()
