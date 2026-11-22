"""Confidence distribution for the locked recipe (top-4 LoRA, single seed).

After training, bin every TEST record by the model's max-softmax confidence and
report, per bin: how many records, what share of the test set, and the accuracy
WITHIN that bin. The accuracy-per-bin is the point -- it shows whether high
confidence actually means high accuracy (the basis for the abstain gate).

Reports both RAW softmax and TEMPERATURE-CALIBRATED confidence (temperature fit
on a held-out calibration split).
"""

import collections, io, json, math, time, urllib.request, zipfile
import torch, torch.nn as nn, torch.nn.functional as F

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
MODEL = "answerdotai/ModernBERT-base"
MAXLEN = 128
CAP = 90
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
CUT = 18
SEED = 11
EPOCHS = 6
LR = 1e-3
R, ALPHA = 16, 32
BINS = [(0.90, 1.01, ">=90%"), (0.70, 0.90, "70-90%"), (0.50, 0.70, "50-70%"), (0.0, 0.50, "<50%")]


def _get(url, timeout=120):
    for a in range(6):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == 5: raise
            time.sleep(10 * (a + 1))


class Passthrough(nn.Module):
    def __init__(self, at): super().__init__(); self.attention_type = at
    def forward(self, h, **kw): return (h,)


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__(); self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.empty(base.in_features, r)); self.B = nn.Parameter(torch.zeros(r, base.out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)); self.scaling = alpha / r
    def forward(self, x): return self.base(x) + (x @ self.A @ self.B) * self.scaling


class Head(nn.Module):
    def __init__(self, d): super().__init__(); self.cls = nn.Linear(d, 6)
    def forward(self, p): return self.cls(p)


def wrap(m, r, alpha):
    for n, c in list(m.named_children()):
        if isinstance(c, nn.Linear): setattr(m, n, LoRALinear(c, r, alpha))
        else: wrap(c, r, alpha)


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    path = snapshot_download(MODEL); tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    data = json.loads(zipfile.ZipFile(io.BytesIO(_get(VCDB_ZIP))).read("vcdb_1-of-1.json"))
    def label(r):
        acts = [a for a in r.get("action", {}) if a in CLASSES]
        return acts[0] if len(acts) == 1 else None
    b = collections.defaultdict(list)
    for r in data:
        lab = label(r); s = r.get("summary")
        if lab in CLASSES and isinstance(s, str) and len(s.strip()) > 40 and len(b[lab]) < CAP:
            b[lab].append(s.strip()[:1000])
    texts, labels = [], []
    for c in CLASSES:
        for s in b[c]: texts.append(s); labels.append(IDX[c])
    n = len(texts); y = torch.tensor(labels)
    E = tok(texts, padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")
    ids, mask = E["input_ids"], E["attention_mask"]
    g = torch.Generator().manual_seed(SEED); perm = torch.randperm(n, generator=g).tolist()
    tr, cal, te = perm[:int(n*.60)], perm[int(n*.60):int(n*.74)], perm[int(n*.74):]
    print(f"records {n} | train {len(tr)} cal {len(cal)} test {len(te)}\n", flush=True)

    base = AutoModel.from_pretrained(path, local_files_only=True).eval()
    for p in base.parameters(): p.requires_grad = False
    D = base.config.hidden_size; cache = torch.zeros(n, MAXLEN, D)
    with torch.no_grad():
        for i in range(0, n, 16):
            cache[i:i+16] = base(input_ids=ids[i:i+16], attention_mask=mask[i:i+16], output_hidden_states=True).hidden_states[CUT]
    del base

    m = AutoModel.from_pretrained(path, local_files_only=True)
    for i in range(CUT): m.layers[i] = Passthrough(m.layers[i].attention_type)
    for p in m.parameters(): p.requires_grad = False
    torch.manual_seed(3)
    for i in range(CUT, len(m.layers)): wrap(m.layers[i], R, ALPHA)
    torch.manual_seed(7); head = Head(D)
    inject = {"buf": None}
    def hook(mod, a, kw): return ((inject["buf"],) + tuple(a[1:]), kw)
    m.layers[CUT].register_forward_pre_hook(hook, with_kwargs=True)

    def logits(idxs):
        inject["buf"] = cache[idxs]
        last = m(input_ids=ids[idxs], attention_mask=mask[idxs]).last_hidden_state
        mm = mask[idxs].unsqueeze(-1).float()
        return head((last*mm).sum(1) / mm.sum(1).clamp(min=1.0))

    params = [p for p in m.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.Adam(params, lr=LR); lf = nn.CrossEntropyLoss()
    m.train(); head.train()
    for ep in range(EPOCHS):
        order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(SEED*9+ep)).tolist()
        for i in range(0, len(tr), 16):
            bi = torch.tensor([tr[order[k]] for k in range(i, min(i+16, len(tr)))])
            opt.zero_grad(); lf(logits(bi), y[bi]).backward(); opt.step()

    m.eval(); head.eval()
    with torch.no_grad():
        lg_cal = logits(torch.tensor(cal)); lg_te = logits(torch.tensor(te))
    y_te = y[torch.tensor(te)]

    # temperature: grid search minimizing NLL on cal
    best_T, best_nll = 1.0, 1e9
    for T in [x/10 for x in range(5, 41)]:
        nll = F.cross_entropy(lg_cal / T, y[torch.tensor(cal)]).item()
        if nll < best_nll: best_nll, best_T = nll, T

    def report(tag, lg, T):
        prob = F.softmax(lg / T, 1); conf, pred = prob.max(1)
        nte = len(y_te); overall = (pred == y_te).float().mean().item()*100
        print(f"--- {tag} (T={T}) | overall acc {overall:.1f}% ---", flush=True)
        for lo, hi, name in BINS:
            sel = (conf >= lo) & (conf < hi)
            k = sel.sum().item()
            if k == 0:
                print(f"  {name:7} {k:3d} records ({0:4.1f}%)   acc    -", flush=True); continue
            acc = (pred[sel] == y_te[sel]).float().mean().item()*100
            print(f"  {name:7} {k:3d} records ({k/nte*100:4.1f}%)   acc {acc:5.1f}%", flush=True)
        print(flush=True)

    report("RAW softmax", lg_te, 1.0)
    report("TEMPERATURE-CALIBRATED", lg_te, best_T)


main()
