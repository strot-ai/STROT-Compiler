"""Code templates baked into the compiled decision model.

The synthesizer LLM generates only the task-specific state projection
(``route_context``); the decision runtime (frozen encoder + one head per
question, optionally a shared LoRA adapter) is rendered deterministically from
these templates so the compiled artifact is fully standalone - zero imports from
``strot`` - and emits typed decision JSON, nothing generated.
"""

from __future__ import annotations

DECIDE_DISTILLED_HEADER_TEMPLATE = '''\
"""Compiled STROT decision model. DO NOT EDIT BY HAND.

Synthesized by strot-ai (STROT: Structured Task Reasoning and Output
Transformation, arXiv:2505.01636) in decide mode. Per record the frozen encoder
produces one embedding and a compile-time-distilled logistic head per question
answers it; output is exclusively type-safe JSON (predicates / scores / choices
with probabilities and confidences) - nothing is generated, only selected.

    base encoder : {nlu_model}

Runtime dependency: pip install strot-ai[local] (torch + transformers).
"""

import json
import math
import os
from typing import Any, Dict, List, Mapping, Sequence

try:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer
except ImportError as _exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "This distilled decision model embeds records with the local encoder: "
        "pip install strot-ai[local]"
    ) from _exc

NLU_MODEL = {nlu_model!r}

DATASET_SCHEMA: Dict[str, Any] = {schema_literal}

#: The original typed-question config this model was compiled from.
QUESTIONS_CONFIG: Dict[str, Any] = {config_literal}
'''

DECIDE_DISTILLED_RUNTIME_TEMPLATE = '''\
# --- Runtime: one compile-time-distilled head per question on the frozen encoder ---
# Each head is a logistic classifier trained at compile time on teacher-labeled
# records (train fold only); weights are baked in QUESTION_HEADS as literals.


def _head_softmax(vector: Sequence[float], weights: Sequence[Sequence[float]],
                  bias: Sequence[float], temperature: float = 1.0) -> List[float]:
    """softmax((W.x + b) / T) - temperature fitted at compile time per head."""
    logits = [(sum(w * x for w, x in zip(row, vector)) + b) / temperature
              for row, b in zip(weights, bias)]
    peak = max(logits)
    exps = [math.exp(z - peak) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _nearest_centroid_distance(vector: Sequence[float],
                               centroids: Sequence[Sequence[float]]) -> float:
    return min(math.sqrt(sum((x - c) * (x - c) for x, c in zip(vector, cen)))
               for cen in centroids)


def _is_ood(vector: Sequence[float]) -> bool:
    """Record farther than the fitted threshold from every class centroid.

    Drift guard: an empty OOD literal (lite, or an unfit gate) is a no-op."""
    return bool(OOD) and _nearest_centroid_distance(vector, OOD["centroids"]) > OOD["threshold"]


#: Which output group each question type lands in.
_DECISION_GROUP = {{"predicate": "predicates", "score": "scores", "choice": "choices"}}


class CompiledDecider:
    """Frozen-encoder embedding + one distilled head per question."""

    def __init__(self) -> None:
        checkpoint = os.environ.get("STROT_NLU_CHECKPOINT")
        if checkpoint is None:
            from pathlib import Path

            if Path(NLU_MODEL).is_dir():
                checkpoint = NLU_MODEL
            else:
                from huggingface_hub import snapshot_download

                checkpoint = snapshot_download(NLU_MODEL)
        if os.path.isdir(os.path.join(checkpoint, "encoder")):
            self._tokenizer = AutoTokenizer.from_pretrained(checkpoint + "/tokenizer")
            config = AutoConfig.from_pretrained(checkpoint + "/encoder")
            self._model = AutoModel.from_config(config)
            state = load_file(checkpoint + "/model.safetensors")
            self._model.load_state_dict(
                {{
                    key[len("encoder."):]: value
                    for key, value in state.items()
                    if key.startswith("encoder.")
                }},
                strict=False,
            )
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self._model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        self._model.eval()

    def _embed(self, text: str) -> List[float]:
        inputs = self._tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True, max_length=256,
        )
        with torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        return ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0))[0].tolist()

    def process(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        vector = self._embed(route_context(dict(record)))  # synthesized projection
        ood = _is_ood(vector)  # drift guard: refuse records unlike the training set
        output: Dict[str, Dict[str, Any]] = {{"predicates": {{}}, "scores": {{}}, "choices": {{}}}}
        for name, head in QUESTION_HEADS.items():
            classes = head["classes"]
            if ood:
                output[_DECISION_GROUP[head["type"]]][name] = {{"refusal": True, "reason": "ood"}}
                continue
            probs = _head_softmax(vector, head["weights"], head["bias"],
                                  head.get("temperature", 1.0))
            if head["type"] == "predicate":
                p_true = probs[classes.index(True)]
                output["predicates"][name] = {{
                    "type": "predicate", "probability": round(float(p_true), 4),
                }}
            elif head["type"] == "score":
                idx = max(range(len(probs)), key=probs.__getitem__)
                output["scores"][name] = {{
                    "score": classes[idx],
                    "probabilities": {{classes[i]: round(float(p), 4)
                                       for i, p in enumerate(probs)}},
                    "confidence": round(float(max(probs)), 4),
                }}
            else:
                idx = max(range(len(probs)), key=probs.__getitem__)
                output["choices"][name] = {{
                    "choice": classes[idx],
                    "probabilities": {{classes[i]: round(float(p), 4)
                                       for i, p in enumerate(probs)}},
                    "confidence": round(float(max(probs)), 4),
                }}
        return output


if __name__ == "__main__":
    import sys

    decider = CompiledDecider()
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(decider.process(json.loads(line)), default=str))
'''

DECIDE_FINETUNED_RUNTIME_TEMPLATE = '''\
# --- Runtime: shared top-N LoRA adapter + one head per question ---
# ADAPTER / ADAPTER_CONFIG hold a cached-lower + top-N LoRA adapter trained
# jointly over all questions at compile time (train fold); each head in
# QUESTION_HEADS was then refit + calibrated on the adapted embedding.
import torch.nn as _nn
from huggingface_hub import snapshot_download


class _PassthroughLayer(_nn.Module):
    def __init__(self, attention_type: Any) -> None:
        super().__init__()
        self.attention_type = attention_type

    def forward(self, hidden_states: Any, **kwargs: Any) -> Any:
        return (hidden_states,)


class _LoRALinear(_nn.Module):
    def __init__(self, inner: Any, r: int, alpha: int) -> None:
        super().__init__()
        self.inner = inner
        self.A = _nn.Parameter(torch.zeros(inner.in_features, r))
        self.B = _nn.Parameter(torch.zeros(r, inner.out_features))
        self.scaling = alpha / r

    def forward(self, x: Any) -> Any:
        return self.inner(x) + (x @ self.A @ self.B) * self.scaling


def _head_softmax(vector: Sequence[float], weights: Sequence[Sequence[float]],
                  bias: Sequence[float], temperature: float = 1.0) -> List[float]:
    """softmax((W.x + b) / T) - temperature fitted at compile time per head."""
    logits = [(sum(w * x for w, x in zip(row, vector)) + b) / temperature
              for row, b in zip(weights, bias)]
    peak = max(logits)
    exps = [math.exp(z - peak) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _nearest_centroid_distance(vector: Sequence[float],
                               centroids: Sequence[Sequence[float]]) -> float:
    return min(math.sqrt(sum((x - c) * (x - c) for x, c in zip(vector, cen)))
               for cen in centroids)


def _is_ood(vector: Sequence[float]) -> bool:
    """Record farther than the fitted threshold from every class centroid."""
    return bool(OOD) and _nearest_centroid_distance(vector, OOD["centroids"]) > OOD["threshold"]


#: Which output group each question type lands in.
_DECISION_GROUP = {{"predicate": "predicates", "score": "scores", "choice": "choices"}}


class CompiledDecider:
    """Shared LoRA-adapted encoder embedding + one head per question."""

    def __init__(self) -> None:
        config = ADAPTER_CONFIG
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        path = snapshot_download(config["base_model"])
        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # float32 to match the baked LoRA weights (the gte encoder may load Half).
        model = AutoModel.from_pretrained(path, trust_remote_code=True).to(self._device).float()
        for p in model.parameters():
            p.requires_grad = False
        n_layers = int(config["n_layers"])
        cut = int(config["cut"])
        self._max_len = int(config["max_length"])

        def _wrap(module: Any) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, _nn.Linear):
                    setattr(module, name, _LoRALinear(
                        child, int(config["lora_rank"]), int(config["lora_alpha"])))
                else:
                    _wrap(child)

        # Serving runs the real frozen lower layers; only top-N carry the adapter.
        for i in range(cut, n_layers):
            _wrap(model.layers[i])
        own = dict(model.named_parameters())
        for name, value in ADAPTER.items():
            if name in own:
                own[name].data = torch.tensor(value, device=self._device)
        model.eval()
        self._model = model

    def _embed(self, text: str) -> List[float]:
        enc = self._tokenizer(
            [text], padding="max_length", truncation=True,
            max_length=self._max_len, return_tensors="pt",
        )
        ids = enc["input_ids"].to(self._device)
        mask = enc["attention_mask"].to(self._device)
        with torch.no_grad():
            last = self._model(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return ((last * m).sum(1) / m.sum(1).clamp(min=1.0))[0].cpu().tolist()

    def process(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        vector = self._embed(route_context(dict(record)))  # synthesized projection
        ood = _is_ood(vector)  # drift guard: refuse records unlike the training set
        output: Dict[str, Dict[str, Any]] = {{"predicates": {{}}, "scores": {{}}, "choices": {{}}}}
        for name, head in QUESTION_HEADS.items():
            classes = head["classes"]
            if ood:
                output[_DECISION_GROUP[head["type"]]][name] = {{"refusal": True, "reason": "ood"}}
                continue
            probs = _head_softmax(vector, head["weights"], head["bias"],
                                  head.get("temperature", 1.0))
            if head["type"] == "predicate":
                p_true = probs[classes.index(True)]
                output["predicates"][name] = {{
                    "type": "predicate", "probability": round(float(p_true), 4),
                }}
            elif head["type"] == "score":
                idx = max(range(len(probs)), key=probs.__getitem__)
                output["scores"][name] = {{
                    "score": classes[idx],
                    "probabilities": {{classes[i]: round(float(p), 4)
                                       for i, p in enumerate(probs)}},
                    "confidence": round(float(max(probs)), 4),
                }}
            else:
                idx = max(range(len(probs)), key=probs.__getitem__)
                output["choices"][name] = {{
                    "choice": classes[idx],
                    "probabilities": {{classes[i]: round(float(p), 4)
                                       for i, p in enumerate(probs)}},
                    "confidence": round(float(max(probs)), 4),
                }}
        return output


if __name__ == "__main__":
    import sys

    decider = CompiledDecider()
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(json.dumps(decider.process(json.loads(line)), default=str))
'''

