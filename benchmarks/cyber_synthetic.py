"""Does LLM-generated synthetic data help cyber? Grounded-in-real + some random.

For each class: show the teacher a few REAL train examples -> generate same-style
messy variations (grounded), PLUS some broader diverse scenarios (random). Add
the synthetic to the real train set, fine-tune top-4 LoRA, and eval on REAL
held-out. Compared to a real-only arm. Real stays the majority; eval is real-only.
"""
import collections, io, json, math, os, time, urllib.request, zipfile
import torch, torch.nn as nn, torch.nn.functional as F

VCDB_ZIP = "https://github.com/vz-risk/VCDB/raw/master/data/joined/vcdb.json.zip"
MODEL = "answerdotai/ModernBERT-base"
MAXLEN = 128; CAP = 90
CLASSES = ["hacking", "malware", "error", "misuse", "physical", "social"]
IDX = {c: i for i, c in enumerate(CLASSES)}
CUT = 18; EPOCHS = 6; LR = 1e-3; R, ALPHA = 16, 32; SEED = 11
N_GROUNDED, N_RANDOM = 16, 10   # synthetic per class (minority vs ~388 real train)
DEFS = {
    "hacking": "unauthorized access by exploiting/bypassing controls (brute force, SQLi, stolen creds, exploits)",
    "malware": "malicious software (virus, ransomware, trojan, backdoor, spyware)",
    "error": "an authorized person's unintentional mistake (misconfig, misdelivery, accidental disclosure, lost device)",
    "misuse": "an authorized insider abusing legitimate access/privileges",
    "physical": "physical action (theft/loss of a device or document, tampering, surveillance)",
    "social": "manipulating a person (phishing, pretexting, bribery, spoofed emails)",
}


def _get(url, timeout=120):
    for a in range(6):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r: return r.read()
        except Exception:
            if a == 5: raise
            time.sleep(10*(a+1))


def gen_synthetic(real_by_class):
    import anthropic
    client = anthropic.Anthropic()
    synth_texts, synth_labels = [], []
    for c in CLASSES:
        examples = "\n".join(f"- {e}" for e in real_by_class[c][:5])
        grounded = (
            f"Here are real cybersecurity incident summaries of type '{c}' ({DEFS[c]}):\n{examples}\n\n"
            f"Write {N_GROUNDED} NEW, realistic incident summaries of the SAME type. Match the real "
            f"style: messy, varied length and phrasing, concrete but not textbook. Invent genuinely new "
            f"scenarios (do NOT paraphrase the examples). One per line, no numbering, no preamble."
        )
        rand = (
            f"Write {N_RANDOM} diverse, realistic cybersecurity incident summaries of type '{c}' "
            f"({DEFS[c]}). Vary the scenarios as widely as possible; real-world messy style. "
            f"One per line, no numbering, no preamble."
        )
        for prompt in (grounded, rand):
            try:
                resp = client.messages.create(
                    model="claude-opus-4-8", max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in resp.content if b.type == "text")
            except Exception as e:
                print(f"  gen fail {c}: {e}", flush=True); continue
            for line in text.split("\n"):
                s = line.strip().lstrip("-*0123456789. ").strip()
                if len(s) > 30:
                    synth_texts.append(s[:1000]); synth_labels.append(IDX[c])
    return synth_texts, synth_labels


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


def cov90(prob, y):
    conf, pred = prob.max(1)
    cs = (pred == y).int()[torch.argsort(conf, descending=True)]
    best = 0
    for k in range(1, len(y)+1):
        if cs[:k].float().mean().item()*100 >= 90: best = k
    return best/len(y)*100


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer
    path = snapshot_download(MODEL); tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    data = json.loads(zipfile.ZipFile(io.BytesIO(_get(VCDB_ZIP))).read("vcdb_1-of-1.json"))
    def label(r):
        acts = [a for a in r.get("action", {}) if a in CLASSES]
        return acts[0] if len(acts) == 1 else None
    buck = collections.defaultdict(list)
    for r in data:
        lab = label(r); s = r.get("summary")
        if lab in CLASSES and isinstance(s, str) and len(s.strip()) > 40 and len(buck[lab]) < CAP:
            buck[lab].append(s.strip()[:1000])
    real_texts, real_labels = [], []
    for c in CLASSES:
        for s in buck[c]: real_texts.append(s); real_labels.append(IDX[c])
    n_real = len(real_texts)
    g = torch.Generator().manual_seed(SEED); perm = torch.randperm(n_real, generator=g).tolist()
    tr, te = perm[:int(n_real*.72)], perm[int(n_real*.72):]

    print("generating synthetic (grounded in real train + random)...", flush=True)
    real_by_class = collections.defaultdict(list)
    for i in tr:
        real_by_class[CLASSES[real_labels[i]]].append(real_texts[i])
    synth_texts, synth_labels = gen_synthetic(real_by_class)
    print(f"real train {len(tr)} | synthetic {len(synth_texts)} | real test {len(te)}", flush=True)
    for c in CLASSES:
        print(f"  {c}: {sum(1 for l in synth_labels if l==IDX[c])} synth", flush=True)

    all_texts = real_texts + synth_texts
    all_labels = real_labels + synth_labels
    synth_idx = list(range(n_real, len(all_texts)))
    y = torch.tensor(all_labels)
    E = tok(all_texts, padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")
    ids, mask = E["input_ids"], E["attention_mask"]

    base = AutoModel.from_pretrained(path, local_files_only=True).eval()
    for p in base.parameters(): p.requires_grad = False
    D = base.config.hidden_size; cache = torch.zeros(len(all_texts), MAXLEN, D)
    with torch.no_grad():
        for i in range(0, len(all_texts), 16):
            cache[i:i+16] = base(input_ids=ids[i:i+16], attention_mask=mask[i:i+16], output_hidden_states=True).hidden_states[CUT]
    del base
    y_te = y[torch.tensor(te)]

    def train_eval(train_ix, seed):
        m = AutoModel.from_pretrained(path, local_files_only=True)
        for i in range(CUT): m.layers[i] = Passthrough(m.layers[i].attention_type)
        for p in m.parameters(): p.requires_grad = False
        torch.manual_seed(seed)
        for i in range(CUT, len(m.layers)): wrap(m.layers[i], R, ALPHA)
        head = nn.Linear(D, 6); inj = {"buf": None}
        m.layers[CUT].register_forward_pre_hook(lambda mo, a, kw: ((inj["buf"],) + tuple(a[1:]), kw), with_kwargs=True)
        def lg(idx):
            inj["buf"] = cache[idx]
            last = m(input_ids=ids[idx], attention_mask=mask[idx]).last_hidden_state
            mm = mask[idx].unsqueeze(-1).float()
            return head((last*mm).sum(1)/mm.sum(1).clamp(min=1.0))
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad] + list(head.parameters()), lr=LR)
        lf = nn.CrossEntropyLoss(); m.train(); head.train(); L = torch.tensor(train_ix)
        for ep in range(EPOCHS):
            o = torch.randperm(len(train_ix), generator=torch.Generator().manual_seed(seed*9+ep)).tolist()
            for i in range(0, len(train_ix), 16):
                bi = L[[o[k] for k in range(i, min(i+16, len(train_ix)))]]
                opt.zero_grad(); lf(lg(bi), y[bi]).backward(); opt.step()
        m.eval(); head.eval()
        with torch.no_grad(): p = F.softmax(lg(torch.tensor(te)), 1)
        acc = (p.argmax(1) == y_te).float().mean().item()*100
        return acc, cov90(p, y_te)

    print("\n=== arms (single top-4 LoRA, eval on REAL held-out) ===", flush=True)
    a_acc, a_cov = train_eval(tr, SEED)
    print(f"REAL-ONLY        ({len(tr)} rec):  acc {a_acc:.1f}%  cov@90 {a_cov:.1f}%", flush=True)
    b_acc, b_cov = train_eval(tr + synth_idx, SEED)
    print(f"REAL+SYNTHETIC   ({len(tr)+len(synth_idx)} rec):  acc {b_acc:.1f}%  cov@90 {b_cov:.1f}%", flush=True)
    print(f"\nDELTA: acc {b_acc-a_acc:+.1f}pp   cov@90 {b_cov-a_cov:+.1f}pp", flush=True)


main()
