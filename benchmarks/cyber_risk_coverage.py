"""Risk-coverage (selective prediction) curve for top-4 LoRA.

Sort test records by confidence (descending) and, for each coverage level, answer
the most-confident X% and report accuracy on the answered set. This frontier is
CALIBRATION-INVARIANT (temperature is monotonic -> same ranking), so it's the true
picture of the model's selective-prediction ability. Shows the coverage/accuracy
trade-off and the max coverage achievable at a 90% / 95% accuracy target.
"""

import collections, io, json, math, time, urllib.request, zipfile
import torch, torch.nn as nn, torch.nn.functional as F

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
MODEL = "answerdotai/ModernBERT-base"
MAXLEN = 128; CAP = 90
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
CUT = 18; SEED = 11; EPOCHS = 6; LR = 1e-3; R, ALPHA = 16, 32


def _get(url, timeout=120):
    for a in range(6):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r: return r.read()
        except Exception:
            if a == 5: raise
            time.sleep(10*(a+1))


class Passthrough(nn.Module):
    def __init__(self, at): super().__init__(); self.attention_type = at
    def forward(self, h, **kw): return (h,)


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__(); self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.empty(base.in_features, r)); self.B = nn.Parameter(torch.zeros(r, base.out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)); self.scaling = alpha/r
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
    tr, te = perm[:int(n*.74)], perm[int(n*.74):]
    print(f"records {n} | train {len(tr)} test {len(te)}\n", flush=True)

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
    m.layers[CUT].register_forward_pre_hook(lambda mod, a, kw: ((inject["buf"],) + tuple(a[1:]), kw), with_kwargs=True)

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
    with torch.no_grad(): lg = logits(torch.tensor(te))
    y_te = y[torch.tensor(te)]
    conf, pred = F.softmax(lg, 1).max(1)
    correct = (pred == y_te).int()
    order = torch.argsort(conf, descending=True)
    correct_sorted = correct[order]
    nte = len(y_te)

    print(f"overall acc {correct.float().mean().item()*100:.1f}%  (answer-everything)\n", flush=True)
    print("coverage |  answered  |  accuracy", flush=True)
    for cov in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        k = max(1, round(nte*cov/100))
        acc = correct_sorted[:k].float().mean().item()*100
        print(f"  {cov:3d}%   |  {k:3d}/{nte}  |   {acc:5.1f}%", flush=True)

    # max coverage at accuracy targets
    print(flush=True)
    for tgt in [95, 90, 85]:
        best = 0
        for k in range(1, nte+1):
            if correct_sorted[:k].float().mean().item()*100 >= tgt: best = k
        print(f"  max coverage @ >={tgt}% acc:  {best/nte*100:.0f}% ({best}/{nte} records)", flush=True)


main()
