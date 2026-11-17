"""Do the two cov@90 levers STACK? At each training-data size (simulating flywheel
growth), train K=3 diverse top-4 LoRA models and compare single-model cov@90 to
the ENSEMBLE cov@90. If cov@90 rises with data AND the ensemble adds on top at
every size, the data lever (flywheel) and the ranking lever (ensemble) stack.
"""

import collections, io, json, math, time, urllib.request, zipfile
import torch, torch.nn as nn, torch.nn.functional as F

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
MODEL = "answerdotai/ModernBERT-base"
MAXLEN = 128; CAP = 90
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
CUT = 18; EPOCHS = 5; LR = 1e-3; R, ALPHA = 16, 32
SEEDS = [11, 23, 37]
SIZES = [60, 150, 250, 399]


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


def wrap(m, r, alpha):
    for n, c in list(m.named_children()):
        if isinstance(c, nn.Linear): setattr(m, n, LoRALinear(c, r, alpha))
        else: wrap(c, r, alpha)


def cov90(prob, y_te):
    conf, pred = prob.max(1)
    cs = (pred == y_te).int()[torch.argsort(conf, descending=True)]
    best = 0
    for k in range(1, len(y_te)+1):
        if cs[:k].float().mean().item()*100 >= 90: best = k
    return best/len(y_te)*100


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
    gp = torch.randperm(n, generator=torch.Generator().manual_seed(11)).tolist()
    trpool, te = gp[:int(n*.74)], gp[int(n*.74):]
    y_te = y[torch.tensor(te)]
    print(f"records {n} | train pool {len(trpool)} test {len(te)} | K={len(SEEDS)} ensemble\n", flush=True)

    base = AutoModel.from_pretrained(path, local_files_only=True).eval()
    for p in base.parameters(): p.requires_grad = False
    D = base.config.hidden_size; cache = torch.zeros(n, MAXLEN, D)
    with torch.no_grad():
        for i in range(0, n, 16):
            cache[i:i+16] = base(input_ids=ids[i:i+16], attention_mask=mask[i:i+16], output_hidden_states=True).hidden_states[CUT]
    del base

    def build(sd):
        m = AutoModel.from_pretrained(path, local_files_only=True)
        for i in range(CUT): m.layers[i] = Passthrough(m.layers[i].attention_type)
        for p in m.parameters(): p.requires_grad = False
        torch.manual_seed(sd)
        for i in range(CUT, len(m.layers)): wrap(m.layers[i], R, ALPHA)
        head = nn.Linear(D, 6); inj = {"buf": None}
        m.layers[CUT].register_forward_pre_hook(lambda mo, a, kw: ((inj["buf"],) + tuple(a[1:]), kw), with_kwargs=True)
        return m, head, inj

    def lg_of(m, head, inj, idxs):
        inj["buf"] = cache[idxs]
        last = m(input_ids=ids[idxs], attention_mask=mask[idxs]).last_hidden_state
        mm = mask[idxs].unsqueeze(-1).float()
        return head((last*mm).sum(1) / mm.sum(1).clamp(min=1.0))

    def train_prob(tr, sd):
        m, head, inj = build(sd)
        params = [p for p in m.parameters() if p.requires_grad] + list(head.parameters())
        opt = torch.optim.Adam(params, lr=LR); lf = nn.CrossEntropyLoss()
        m.train(); head.train(); L = torch.tensor(tr)
        for ep in range(EPOCHS):
            o = torch.randperm(len(tr), generator=torch.Generator().manual_seed(sd*9+ep)).tolist()
            for i in range(0, len(tr), 16):
                bi = L[[o[k] for k in range(i, min(i+16, len(tr)))]]
                opt.zero_grad(); lf(lg_of(m, head, inj, bi), y[bi]).backward(); opt.step()
        m.eval(); head.eval()
        with torch.no_grad(): p = F.softmax(lg_of(m, head, inj, torch.tensor(te)), 1)
        del m, head; return p

    print(f"{'size':>5} | {'single cov@90':>13} | {'ensemble cov@90':>15} | gain", flush=True)
    for sz in SIZES:
        tr = trpool[:sz]
        probs = [train_prob(tr, sd) for sd in SEEDS]
        single = sum(cov90(p, y_te) for p in probs)/len(probs)
        ens = cov90(torch.stack(probs).mean(0), y_te)
        print(f"{sz:>5} | {single:>12.1f}% | {ens:>14.1f}% | {ens-single:+.1f}pp", flush=True)


main()
