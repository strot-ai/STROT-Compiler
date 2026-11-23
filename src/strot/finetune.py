"""Compile-time fine-tune tier: the optimizer rung above frozen distillation.

:mod:`strot.distill` trains a logistic head on the encoder's frozen
embeddings, which caps at the frozen representation ceiling. For
representation-bound tasks - where the decision-critical distinction is not
linearly present in the pooled embedding - no head, retrieval, or reframing
crosses that ceiling; only changing weights does. This module is that change,
kept as cheap and CPU-survivable as possible.

The lower encoder layers are frozen and their activations are computed once
and cached; only the top ``N`` layers carry a trainable LoRA adapter. Training
backprops through the top block only, so it finishes on CPU in minutes and
sidesteps the memory blow-up of full-encoder fine-tuning. ``N`` is small by
design (top-4 is the measured knee); depth beyond that buys noise. Unlike the
pure-Python frozen head this needs torch + transformers (the ``[local]``
extra) and trains an adapter (MB of weights). Those weights are then baked
into the battery ``.py`` as ``ADAPTER`` / ``ADAPTER_CONFIG`` literals (next to
the distilled head, labels, calibration, and OOD centroids), so the battery
stays self-contained except for the shared base-encoder checkpoint it loads at
runtime. The optimizer only reaches this rung when ``optimize=True`` and the
frozen tier misses the target on the real measurement fold.

Fold hygiene is identical to distillation: teacher/synthetic data may TEACH
(train fold); only held-out real records JUDGE (measure fold decides whether
this tier ships; calibrate fold feeds :mod:`strot.calibrate`). This module
returns the trained adapter plus the probability distributions it produces on
the measure and calibrate folds, so the existing temperature / conformal /
selective-cutoff machinery applies unchanged.

Device-agnostic: runs on GPU when ``torch.cuda.is_available()`` (compile in
seconds), else CPU (minutes) - the same code, selected at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence

from .constants import DEFAULT_NLU_MODEL
from .exceptions import RouterConfigurationError

#: Default fine-tune base: the SAME encoder `head` uses, so `deep` is a true
#: superset (head's frozen encoder + a LoRA adapter), not a different, weaker
#: model. gte-modernbert-base loads as a plain ``ModernBertModel`` (identical
#: structure to answerdotai/ModernBERT-base: ``.layers`` + ``.attention_type``),
#: so the hand-rolled LoRA path below works on it unchanged - it just starts from
#: the GTE-tuned weights instead of the raw ModernBERT ones.
DEFAULT_FINETUNE_BASE = DEFAULT_NLU_MODEL

#: Validated recipe defaults (see benchmarks/cyber_cached_top.py,
#: cyber_soup_lora.py). ``top_layers`` is the measured knee.
DEFAULT_TOP_LAYERS = 4
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_EPOCHS = 6
DEFAULT_LR = 1e-3
DEFAULT_MAX_LEN = 128

MIN_FINETUNE_RECORDS = 48  #: below this the folds cannot train a top-N adapter


#: Default ensemble width. The serving router already averages K adapters; the
#: measured +~18pp cov@90 win comes from training a small battery of diverse
#: members rather than a single adapter. Internal knob - never surfaced on the
#: public run()/decide() API; the compiler sets it when it escalates.
DEFAULT_ENSEMBLE_SIZE = 3


@dataclass
class FineTuneConfig:
    """Knobs for the fine-tune tier. Defaults are the validated recipe."""

    base_model: str = DEFAULT_FINETUNE_BASE
    top_layers: int = DEFAULT_TOP_LAYERS
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LR
    max_length: int = DEFAULT_MAX_LEN
    seed: int = 11
    #: Number of diverse adapters to train and average (K). Each member varies
    #: the torch seed (seed, seed+1, ...); their measure/calibrate softmax rows
    #: are averaged for judging + calibration and the list of adapters is baked
    #: into the artifact (serving averages them). 1 == single-adapter behaviour.
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE


def _import_torch():
    try:
        import torch  # noqa: F401
        from huggingface_hub import snapshot_download  # noqa: F401
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env guard
        raise RouterConfigurationError(
            "The fine-tune tier needs torch + transformers: "
            'pip install "strot-ai[local]"'
        ) from exc


@dataclass
class DecideAdaptResult:
    """A shared LoRA adapter plus the adapted embeddings it produces.

    ``adapter`` is the trainable LoRA state (nested python floats, no head - the
    decide heads are refit outside on ``vectors``). ``vectors`` is one adapted
    pooled embedding per input record, in input order, ready for the same
    per-question head-train + calibrate path the frozen tier uses. ``config``
    carries what the runtime needs to rebuild the adapter forward.
    """

    adapter: Dict[str, Any]
    vectors: List[List[float]]
    config: Dict[str, Any] = field(default_factory=dict)


def finetune_decide(
    texts: Sequence[str],
    question_labels: Dict[str, Sequence[int]],
    question_classes: Dict[str, int],
    train: Sequence[int],
    config: Optional[FineTuneConfig] = None,
) -> DecideAdaptResult:
    """Train ONE shared cached-lower + top-N LoRA adapter for all questions.

    The adapter is trained multi-task: a temporary head per question provides the
    gradient (loss = sum of per-question cross-entropies on the ``train`` fold),
    so the shared encoder adapts to the decision domain in a single pass. The
    temporary heads are then discarded; the adapted pooled embedding for every
    record is returned so the caller refits + calibrates the real decide heads on
    the adapted representation (one forward pass per record at serving, N heads).

    The LoRA adapter is initialized fresh every call (A kaiming, B zeros) from
    the frozen base - part of decide's recompile-don't-increment invariant (see
    ``StrotCompiler.compile_decide``). Do not thread in a prior adapter to
    continue training on new-data-only; recompile on the accumulated data so the
    fit stays forgetting-free.
    """
    _import_torch()
    import torch
    import torch.nn as nn

    cfg = config or FineTuneConfig()
    if len(texts) < MIN_FINETUNE_RECORDS:
        raise ValueError(
            f"need >= {MIN_FINETUNE_RECORDS} records to fine-tune; got {len(texts)}"
        )
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = snapshot_download(cfg.base_model)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    enc = tok(
        list(texts), padding="max_length", truncation=True,
        max_length=cfg.max_length, return_tensors="pt",
    )
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)

    # Force float32: gte-modernbert may carry a reduced-precision torch_dtype in
    # its config, and the CPU LoRA path (float32 cache + heads) needs it uniform.
    base = AutoModel.from_pretrained(path, trust_remote_code=True).to(device).float().eval()
    for p in base.parameters():
        p.requires_grad = False
    n_layers = len(base.layers)
    cut = max(0, n_layers - cfg.top_layers)
    dim = base.config.hidden_size

    cache = torch.zeros(len(texts), cfg.max_length, dim, device=device)
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            out = base(input_ids=ids[i:i + 16], attention_mask=mask[i:i + 16],
                       output_hidden_states=True)
            cache[i:i + 16] = out.hidden_states[cut]
    del base

    class _Passthrough(nn.Module):
        def __init__(self, attention_type):
            super().__init__()
            self.attention_type = attention_type

        def forward(self, hidden_states, **kwargs):
            return (hidden_states,)

    class _LoRALinear(nn.Module):
        def __init__(self, inner, r, alpha):
            super().__init__()
            self.inner = inner
            for p in self.inner.parameters():
                p.requires_grad = False
            self.A = nn.Parameter(torch.empty(inner.in_features, r))
            self.B = nn.Parameter(torch.zeros(r, inner.out_features))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.scaling = alpha / r

        def forward(self, x):
            return self.inner(x) + (x @ self.A @ self.B) * self.scaling

    def _wrap(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, _LoRALinear(child, cfg.lora_rank, cfg.lora_alpha))
            else:
                _wrap(child)

    torch.manual_seed(cfg.seed)
    model = AutoModel.from_pretrained(path, trust_remote_code=True).to(device).float()
    for p in model.parameters():
        p.requires_grad = False
    for i in range(cut):
        model.layers[i] = _Passthrough(model.layers[i].attention_type)
    for i in range(cut, n_layers):
        _wrap(model.layers[i])
    names = list(question_classes)
    heads = nn.ModuleDict(
        {name: nn.Linear(dim, question_classes[name]).to(device) for name in names}
    )
    ys = {name: torch.tensor(list(question_labels[name]), device=device) for name in names}
    inject: Dict[str, Any] = {"buf": None}

    def _hook(_module, args, kwargs):
        return ((inject["buf"],) + tuple(args[1:]), kwargs)

    handle = model.layers[cut].register_forward_pre_hook(_hook, with_kwargs=True)

    def _pooled(idx):
        inject["buf"] = cache[idx]
        last = model(input_ids=ids[idx], attention_mask=mask[idx]).last_hidden_state
        m = mask[idx].unsqueeze(-1).float()
        return (last * m).sum(1) / m.sum(1).clamp(min=1.0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    for head in heads.values():
        trainable += list(head.parameters())
    opt = torch.optim.Adam(trainable, lr=cfg.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_idx = torch.tensor(list(train), device=device)
    model.train()
    for head in heads.values():
        head.train()
    for ep in range(cfg.epochs):
        order = torch.randperm(
            len(train_idx), generator=torch.Generator().manual_seed(cfg.seed * 9 + ep)
        )
        for i in range(0, len(train_idx), 16):
            batch = train_idx[order[i:i + 16]]
            opt.zero_grad()
            pooled = _pooled(batch)
            loss = sum(loss_fn(heads[name](pooled), ys[name][batch]) for name in names)
            loss.backward()
            opt.step()

    model.eval()
    vectors: List[List[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = torch.arange(i, min(i + 16, len(texts)), device=device)
            vectors.extend(_pooled(batch).cpu().tolist())
    handle.remove()

    adapter = {
        name: p.detach().cpu().tolist()
        for name, p in model.named_parameters() if p.requires_grad
    }
    return DecideAdaptResult(
        adapter=adapter,
        vectors=vectors,
        config={
            "base_model": cfg.base_model, "top_layers": cfg.top_layers,
            "cut": cut, "n_layers": n_layers, "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha, "max_length": cfg.max_length,
            "device": device,
        },
    )


