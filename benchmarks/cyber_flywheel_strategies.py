"""Flywheel acquisition-strategy bake-off: which harvest strategy is most
label-efficient? Same active-learning loop (top-4 LoRA, seed 60, +40/round x5),
four strategies compared on a fixed test set (accuracy + cov@>=90%).

  random      : random pool records (baseline)
  uncertainty : lowest max-softmax confidence (backfired last time)
  diversity   : k-center greedy on frozen embeddings (spread-out / representative)
  hybrid      : smart buffer -> top-3B uncertain, then k-center diverse within them
"""

import collections, io, json, math, time, urllib.request, zipfile
import torch, torch.nn as nn, torch.nn.functional as F

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
MODEL = "answerdotai/ModernBERT-base"
MAXLEN = 128; CAP = 90
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
CUT = 18; EPOCHS = 4; LR = 1e-3; R, ALPHA = 16, 32
SEED_SIZE = 60; BATCH = 40; ROUNDS = 5


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


def kcenter(cand_emb, seed_emb, k):
    md = torch.cdist(cand_emb, seed_emb).min(1).values
    picks = []
    for _ in range(min(k, cand_emb.shape[0])):
        j = int(md.argmax().item()); picks.append(j)
        md = torch.minimum(md, torch.cdist(cand_emb, cand_emb[j:j+1]).squeeze(1)); md[j] = -1
    return picks


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
    te = gp[:140]; poolall = gp[140:]; seed = poolall[:SEED_SIZE]; pool0 = poolall[SEED_SIZE:]
    y_te = y[torch.tensor(te)]
    print(f"records {n} | test {len(te)} | seed {len(seed)} pool {len(pool0)} | +{BATCH} x{ROUNDS}\n", flush=True)

    base = AutoModel.from_pretrained(path, local_files_only=True).eval()
    for p in base.parameters(): p.requires_grad = False
    D = base.config.hidden_size; cache = torch.zeros(n, MAXLEN, D)
    with torch.no_grad():
        for i in range(0, n, 16):
            cache[i:i+16] = base(input_ids=ids[i:i+16], attention_mask=mask[i:i+16], output_hidden_states=True).hidden_states[CUT]
    del base
    # per-record pooled frozen embedding for diversity distance
    mm = mask.unsqueeze(-1).float(); emb = (cache*mm).sum(1) / mm.sum(1).clamp(min=1.0)
    emb = F.normalize(emb, dim=1)
    print("cached.\n", flush=True)

    def build():
        m = AutoModel.from_pretrained(path, local_files_only=True)
        for i in range(CUT): m.layers[i] = Passthrough(m.layers[i].attention_type)
        for p in m.parameters(): p.requires_grad = False
        torch.manual_seed(3)
        for i in range(CUT, len(m.layers)): wrap(m.layers[i], R, ALPHA)
        torch.manual_seed(7); head = nn.Linear(D, 6)
        inj = {"buf": None}
        m.layers[CUT].register_forward_pre_hook(lambda mo, a, kw: ((inj["buf"],) + tuple(a[1:]), kw), with_kwargs=True)
        return m, head, inj

    def lg_of(m, head, inj, idxs):
        inj["buf"] = cache[idxs]
        last = m(input_ids=ids[idxs], attention_mask=mask[idxs]).last_hidden_state
        mmm = mask[idxs].unsqueeze(-1).float()
        return head((last*mmm).sum(1) / mmm.sum(1).clamp(min=1.0))

    def train_eval(labeled):
        m, head, inj = build()
        params = [p for p in m.parameters() if p.requires_grad] + list(head.parameters())
        opt = torch.optim.Adam(params, lr=LR); lf = nn.CrossEntropyLoss()
        m.train(); head.train(); L = torch.tensor(labeled)
        for ep in range(EPOCHS):
            o = torch.randperm(len(labeled), generator=torch.Generator().manual_seed(ep)).tolist()
            for i in range(0, len(labeled), 16):
                bi = L[[o[k] for k in range(i, min(i+16, len(labeled)))]]
                opt.zero_grad(); lf(lg_of(m, head, inj, bi), y[bi]).backward(); opt.step()
        m.eval(); head.eval()
        with torch.no_grad(): lg = lg_of(m, head, inj, torch.tensor(te))
        conf, pred = F.softmax(lg, 1).max(1)
        acc = (pred == y_te).float().mean().item()*100
        cs = (pred == y_te).int()[torch.argsort(conf, descending=True)]
        cov90 = max([0]+[k for k in range(1, len(y_te)+1) if cs[:k].float().mean().item()*100 >= 90])/len(y_te)*100
        return acc, cov90, m, head, inj

    def select(strat, pool, labeled, m, head, inj, rng):
        if strat == "random":
            idx = torch.randperm(len(pool), generator=rng)[:BATCH].tolist()
            return [pool[i] for i in idx]
        P = torch.tensor(pool)
        if strat in ("uncertainty", "hybrid"):
            m.eval(); head.eval()
            with torch.no_grad(): conf = F.softmax(lg_of(m, head, inj, P), 1).max(1).values
        if strat == "uncertainty":
            return [pool[i] for i in torch.argsort(conf)[:BATCH].tolist()]
        if strat == "diversity":
            loc = kcenter(emb[P], emb[torch.tensor(labeled)], BATCH)
            return [pool[i] for i in loc]
        if strat == "hybrid":
            unc = torch.argsort(conf)[:3*BATCH]           # most-uncertain candidates
            cand = P[unc]
            loc = kcenter(emb[cand], emb[torch.tensor(labeled)], BATCH)
            return [int(cand[i].item()) for i in loc]

    for strat in ("random", "uncertainty", "diversity", "hybrid"):
        labeled = list(seed); pool = list(pool0); rng = torch.Generator().manual_seed(123)
        line = f"{strat:11}"
        for rd in range(ROUNDS):
            acc, cov90, m, head, inj = train_eval(labeled)
            line += f" | {len(labeled)}:{acc:.0f}/{cov90:.0f}"
            if rd < ROUNDS-1 and pool:
                pick = select(strat, pool, labeled, m, head, inj, rng)
                labeled += pick; ps = set(pick); pool = [p for p in pool if p not in ps]
        print(line + "   (labeled:acc/cov@90)", flush=True)


main()
