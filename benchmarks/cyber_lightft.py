"""Light fine-tune vs frozen: does unfreezing just the top 2 layers + head move
cyber accuracy AND the Path-3 answered slice past the frozen 30%?

  frozen     : encoder locked, train only a linear head  (= the ~59% probe)
  light-FT   : unfreeze top 2 transformer layers + head   (parameter-efficient)

Both measured the same way: accuracy on test, then a conformal triage gate
(temp-scale on cal, cutoff bounding wrong-dispatch <= alpha) -> answered% and
acc@answered on held-out test. Real labels judge. 2 seeds (fine-tuning is slow).
Reference: frozen roberta-base ~59% / ~30% answered at alpha=0.05.
"""

import collections
import io
import json
import time
import urllib.request
import zipfile

import torch

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
MODEL = "FacebookAI/roberta-base"
CAP = 90
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
SEEDS = [11, 23]
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


def load_data():
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
            texts.append(s[:1500]); labels.append(IDX[c])
    return texts, labels, {c: len(buckets[c]) for c in CLASSES}


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


def gate_report(tag, conf_cal, pred_cal, y_cal, conf_te, pred_te, y_te):
    nt = len(y_te)
    acc = sum(1 for j in range(nt) if pred_te[j] == y_te[j]) / nt * 100
    line = f"  {tag}: top1 {round(acc,1)}%"
    for a in ALPHAS:
        cut = find_cutoff(conf_cal, pred_cal, y_cal, len(y_cal), a)
        disp = [j for j in range(nt) if conf_te[j] >= cut]
        corr = sum(1 for j in disp if pred_te[j] == y_te[j])
        ans = len(disp) / nt * 100
        acca = corr / len(disp) * 100 if disp else 0.0
        line += f"  |  a={a}: answered {round(ans,1)}% @ {round(acca,1)}%"
    return line, acc


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)

    texts, labels, counts = load_data()
    n = len(texts)
    print(f"records: {n}  {counts}\n", flush=True)

    def encode(batch):
        return tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")

    def conf_pred(model, idxs, T=1.0):
        model.eval()
        conf, pred = [], []
        with torch.no_grad():
            for i in range(0, len(idxs), 16):
                enc = encode([texts[j] for j in idxs[i:i+16]])
                lg = model(**enc).logits / T
                p = torch.softmax(lg, dim=1)
                c, pr = p.max(1)
                conf += c.tolist(); pred += pr.tolist()
        return conf, pred

    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        tr, cal, te = perm[:int(n*.5)], perm[int(n*.5):int(n*.72)], perm[int(n*.72):]
        y_cal, y_te = [labels[i] for i in cal], [labels[i] for i in te]
        print(f"--- seed {seed}  (train {len(tr)} / cal {len(cal)} / test {len(te)}) ---", flush=True)

        # LIGHT FINE-TUNE: unfreeze top 2 layers + classifier
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=6, local_files_only=True)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        for p in model.roberta.encoder.layer[-2:].parameters():
            p.requires_grad = True
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        lf = torch.nn.CrossEntropyLoss()
        model.train()
        for ep in range(EPOCHS):
            order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed*10+ep)).tolist()
            for i in range(0, len(tr), 16):
                bi = [tr[order[k]] for k in range(i, min(i+16, len(tr)))]
                enc = encode([texts[j] for j in bi])
                opt.zero_grad()
                lf(model(**enc).logits, torch.tensor([labels[j] for j in bi])).backward()
                opt.step()
        c_cal, p_cal = conf_pred(model, cal)
        c_te, p_te = conf_pred(model, te)
        line, _ = gate_report("light-FT", c_cal, p_cal, y_cal, c_te, p_te, y_te)
        print(line, flush=True)


main()
