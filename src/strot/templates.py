"""Code templates baked into the compiled production asset.

The synthesizer LLM generates only the task-specific *execution paths*;
the runtime router architecture is rendered deterministically from these
templates so the compiled artifact's routing behavior is guaranteed, fully
standalone (zero imports from ``strot``), and matched exactly to the
``nlu_model`` chosen at compile time.
"""

from __future__ import annotations

ASSET_HEADER_TEMPLATE = '''\
"""Compiled STROT execution asset. DO NOT EDIT BY HAND.

Synthesized by strot-ai (STROT: Structured Task Reasoning and
Output Transformation, arXiv:2505.01636).

    nlu_model : {nlu_model}
    runtime   : {runtime_kind}
    goal      : {goal}

This file is completely self-contained (zero strot-package imports) and
can be dropped into any production pipeline as-is.
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

NLU_MODEL = {nlu_model!r}
CONFIDENCE_THRESHOLD = {threshold}
MAX_FLAT_OPTIONS = 20

#: Fitted at compile time from teacher-labeled sample records (None = uncalibrated).
TEMPERATURE: float = {temperature}
CONFORMAL_Q = {conformal_q}
CALIBRATION: Dict[str, Any] = {calibration_literal}
CLASS_BIAS = {class_bias}
DISPATCH_CUTOFF = {dispatch_cutoff}
POLICY: Dict[str, Any] = {policy_literal}

DATASET_SCHEMA: Dict[str, Any] = {schema_literal}


class FallbackException(Exception):
    """Low-confidence routing refusal: do not execute any path blindly."""

    def __init__(self, message: str, confidence: float, threshold: float) -> None:
        super().__init__(message)
        self.confidence = confidence
        self.threshold = threshold


@dataclass(frozen=True)
class RouteDecision:
    index: int
    option: str
    confidence: float
    distribution: Dict[str, float] = field(default_factory=dict)
'''

LOCAL_RUNTIME_TEMPLATE = '''\
# --- Runtime architecture: Implementation Alpha (local Julia decision model) ---
# Julia is NOT a Transformers AutoModel; it ships its own inference package
# ("supersonic-julia") inside the checkpoint repository.
try:
    from julia import load_model
except ImportError as _exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "This compiled asset routes locally with {nlu_model} and needs the "
        "'supersonic-julia' package (do NOT `pip install julia` - that is an "
        "unrelated package). Install:\\n"
        "  python -c \\"from huggingface_hub import snapshot_download; "
        "snapshot_download('{nlu_model}', local_dir='Julia-1')\\"\\n"
        "  pip install ./Julia-1"
    ) from _exc


class _RuntimeRouter:
    """Resident Julia decision engine over {nlu_model} on CPU."""

    def __init__(self) -> None:
        checkpoint = os.environ.get("STROT_NLU_CHECKPOINT")
        if checkpoint is None:
            from pathlib import Path

            if Path(NLU_MODEL).is_dir():
                checkpoint = NLU_MODEL
            else:
                from huggingface_hub import snapshot_download

                checkpoint = snapshot_download(NLU_MODEL)
        self._engine = load_model(checkpoint, device="cpu")

    def select(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        if len(options) == 1:  # Julia accepts 2-20 options; one action is trivial
            return RouteDecision(0, options[0], 1.0, {{options[0]: 1.0}})
        if len(options) <= MAX_FLAT_OPTIONS:
            return self._select_flat(context, question, options)
        return self._select_hierarchical(context, question, options)

    def _select_hierarchical(
        self, context: str, question: str, options: Sequence[str]
    ) -> RouteDecision:
        # Option Tree Splitter: Julia-1 accuracy degrades beyond 20 choices,
        # so oversized registries route through a multi-tier decision tree.
        n_groups = math.ceil(len(options) / MAX_FLAT_OPTIONS)
        size = math.ceil(len(options) / n_groups)
        groups = [list(options[i : i + size]) for i in range(0, len(options), size)]
        labels = [
            "Options {{0}}-{{1}}: {{2}} ... {{3}}".format(
                list(options).index(g[0]) + 1, list(options).index(g[-1]) + 1, g[0], g[-1]
            )
            for g in groups
        ]
        branch = self.select(
            context, question + " (choose the group containing the best option)", labels
        )
        leaf = self.select(context, question, groups[branch.index])
        winner = groups[branch.index][leaf.index]
        return RouteDecision(
            index=list(options).index(winner),
            option=winner,
            confidence=branch.confidence * leaf.confidence,
            distribution=dict(leaf.distribution),
        )

    def _select_flat(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        result = self._engine.predict(
            rows=[{{"state": context, "question": question, "options": list(options)}}]
        )[0]
        index = int(result["index"])
        probs = [float(p) for p in result["probabilities"]]
        return RouteDecision(
            index=index,
            option=options[index],
            confidence=probs[index],
            distribution=dict(zip(options, probs)),
        )
'''

DISTILLED_RUNTIME_TEMPLATE = '''\
# --- Runtime architecture: compile-time-distilled head on the frozen encoder ---
# The logistic head below was trained at compile time on teacher-labeled real
# records (train fold only); its weights are baked in as literals.
try:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer
except ImportError as _exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "This distilled asset embeds records with the local encoder: "
        "pip install strot-ai[local]"
    ) from _exc


class _RuntimeRouter:
    """Frozen encoder embeddings + distilled logistic head."""

    def __init__(self) -> None:
        checkpoint = os.environ.get("STROT_NLU_CHECKPOINT")
        if checkpoint is None:
            from pathlib import Path

            if Path(NLU_MODEL).is_dir():
                checkpoint = NLU_MODEL
            else:
                from huggingface_hub import snapshot_download

                checkpoint = snapshot_download(NLU_MODEL)
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
        self._model.eval()

    def select(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        inputs = self._tokenizer(
            [context], return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        )
        with torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0))[0]
        vector = pooled.tolist()
        logits = [
            sum(w * x for w, x in zip(row, vector)) + b
            for row, b in zip(HEAD_WEIGHTS, HEAD_BIAS)
        ]
        peak = max(logits)
        exps = [math.exp(z - peak) for z in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        index = max(range(len(probs)), key=probs.__getitem__)
        return RouteDecision(
            index=index,
            option=options[index],
            confidence=probs[index],
            distribution=dict(zip(options, probs)),
        )
'''

HOSTED_RUNTIME_TEMPLATE = '''\
# --- Runtime architecture: Implementation Beta (hosted cloud API) ---
# Deliberately torch/transformers-free to keep the container footprint tiny.
from openai import OpenAI

_PROVIDER, _, _MODEL_ID = NLU_MODEL.partition("/")
_ENV_KEYS = {{
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}}

_RESPONSE_SCHEMA: Dict[str, Any] = {{
    "name": "route_decision",
    "strict": True,
    "schema": {{
        "type": "object",
        "properties": {{
            "selected_index": {{"type": "integer"}},
            "confidence": {{"type": "number"}},
        }},
        "required": ["selected_index", "confidence"],
        "additionalProperties": False,
    }},
}}


class _RuntimeRouter:
    """Structured-Outputs router over {nlu_model}: strict integer index only."""

    def __init__(self) -> None:
        env_key = _ENV_KEYS.get(_PROVIDER, "OPENAI_API_KEY")
        api_key = os.environ.get(env_key)
        if not api_key:
            raise RuntimeError("Missing {{0}} in environment.".format(env_key))
        self._client = OpenAI(api_key=api_key)
        effort = os.environ.get("STROT_NLU_REASONING_EFFORT",
                                "low" if _MODEL_ID == "gpt-6-luna" else None) if _PROVIDER == "openai" else None
        self._request_options: Dict[str, Any] = {{"temperature": 0.0}} if effort in (None, "none") else {{}}
        if effort is not None:
            self._request_options["extra_body"] = {{"reasoning_effort": effort}}

    def select(self, context: str, question: str, options: Sequence[str]) -> RouteDecision:
        numbered = "\\n".join("{{0}}: {{1}}".format(i, opt) for i, opt in enumerate(options))
        response = self._client.chat.completions.create(
            model=_MODEL_ID,
            messages=[
                {{
                    "role": "system",
                    "content": (
                        "You are a deterministic router. Given a context, a question, and a "
                        "zero-indexed option list, respond with the single best option index "
                        "and your confidence in [0, 1]."
                    ),
                }},
                {{
                    "role": "user",
                    "content": "Context: {{0}}\\nQuestion: {{1}}\\nOptions:\\n{{2}}".format(
                        context, question, numbered
                    ),
                }},
            ],
            response_format={{"type": "json_schema", "json_schema": _RESPONSE_SCHEMA}},
            **self._request_options,
        )
        payload = json.loads(response.choices[0].message.content)
        index = int(payload["selected_index"])
        if not 0 <= index < len(options):
            raise FallbackException(
                "Hosted router returned out-of-range index {{0}}.".format(index),
                confidence=0.0,
                threshold=CONFIDENCE_THRESHOLD,
            )
        confidence = max(0.0, min(1.0, float(payload["confidence"])))
        return RouteDecision(
            index=index,
            option=options[index],
            confidence=confidence,
            distribution={{options[index]: confidence}},
        )
'''

DECIDE_HEADER_TEMPLATE = '''\
"""Compiled STROT decision asset. DO NOT EDIT BY HAND.

Synthesized by strot-ai (STROT: Structured Task Reasoning and
Output Transformation, arXiv:2505.01636) in decide mode: a
typed-question config compiled against the verified schema. Per record, the
local decision model answers every question in one batched pass and the
output is exclusively type-safe JSON (booleans / scores / choices with
probabilities and confidences) - nothing is generated, only selected,
scored, and computed.

    nlu_model : {nlu_model}

Runtime dependency: the supersonic-julia package + checkpoint
(see strot.prefetch_nlu; never `pip install julia`).
"""

import json
import os
from typing import Any, Dict, List, Mapping, Sequence

try:
    from julia import load_model
except ImportError as _exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "This compiled decision asset needs the 'supersonic-julia' package "
        "(do NOT `pip install julia`). Install:\\n"
        "  python -c \\"from huggingface_hub import snapshot_download; "
        "snapshot_download('{nlu_model}', local_dir='Julia-1')\\"\\n"
        "  pip install ./Julia-1"
    ) from _exc

NLU_MODEL = {nlu_model!r}

DATASET_SCHEMA: Dict[str, Any] = {schema_literal}

#: The original typed-question config this asset was compiled from.
QUESTIONS_CONFIG: Dict[str, Any] = {config_literal}
'''

DECIDE_FOOTER_TEMPLATE = '''\
# --- Statically synthesized question expansion (STROT synthesizer output) ---
{generated_paths}


def _julia_questions() -> Dict[str, Any]:
    """Expand QUESTIONS into the decision model's native typed format.

    STROT's public type "boolean" maps to the model package's internal
    wire type at this boundary only; it never appears in STROT's API.
    """
    prepared: Dict[str, Any] = {{}}
    for name, spec in QUESTIONS.items():
        question: Dict[str, Any] = {{
            "type": "noul" if spec["type"] == "boolean" else spec["type"],
            "instructions": spec["instructions"],
        }}
        if spec["type"] == "score":
            question["criteria"] = list(spec["rubric"])
        else:
            question["criteria"] = spec["criteria"]
        prepared[name] = question
    return prepared


class CompiledDecider:
    """Self-decision asset: one batched typed-decision pass per record."""

    def __init__(self) -> None:
        checkpoint = os.environ.get("STROT_NLU_CHECKPOINT")
        if checkpoint is None:
            from pathlib import Path

            if Path(NLU_MODEL).is_dir():
                checkpoint = NLU_MODEL
            else:
                from huggingface_hub import snapshot_download

                checkpoint = snapshot_download(NLU_MODEL)
        self._engine = load_model(checkpoint, device="cpu")
        self._questions = _julia_questions()

    def process(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        state = route_context(dict(record))  # compile-time-synthesized projection
        answers = self._engine.predict(state=state, questions=self._questions)["answers"]
        output: Dict[str, Dict[str, Any]] = {{"booleans": {{}}, "scores": {{}}, "choices": {{}}}}
        for name, answer in answers.items():
            if answer["type"] == "noul":  # model package's wire name for boolean
                output["booleans"][name] = {{
                    "type": "boolean",
                    "probability": round(float(answer["noul"]), 4),
                }}
            elif answer["type"] == "score":
                value = QUESTIONS[name]["min"] + float(answer["score"])
                output["scores"][name] = {{
                    "value": round(value, 2),
                    "confidence": round(float(answer["max_probability"]), 4),
                }}
            else:
                output["choices"][name] = {{
                    "selected": answer["choice"],
                    "probabilities": {{
                        key: round(float(p), 4)
                        for key, p in answer["probabilities"].items()
                    }},
                    "confidence": round(float(answer["max_probability"]), 4),
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

ASSET_FOOTER_TEMPLATE = '''\
# --- Statically generated execution paths (STROT synthesizer output) ---
{generated_paths}

#: Ordered action registry: list position == routing option index.
ACTION_REGISTRY: Dict[str, str] = {registry_literal}

_HANDLERS = {handlers_literal}

#: The decision model sees ONLY these compile-time-synthesized labels;
#: short discriminative labels measurably outperform verbose descriptions.
_ROUTING_OPTIONS: List[str] = [ROUTING_LABELS[key] for key in ACTION_REGISTRY]
_ACTION_KEYS: List[str] = list(ACTION_REGISTRY)


class CompiledPipeline:
    """Self-routing execution asset: route each record, enforce confidence,
    dispatch to the statically compiled path."""

    def __init__(self) -> None:
        self._router = _RuntimeRouter()

    def process(
        self,
        record: Mapping[str, Any],
        question: str = {default_question!r},
    ) -> Dict[str, Any]:
        context = route_context(dict(record))  # compile-time-synthesized projection
        decision = self._router.select(context, question, _ROUTING_OPTIONS)
        # Inference-level correction (fitted at compile): temperature +
        # per-class bias over the arbiter's distribution.
        index = decision.index
        confidence = decision.confidence
        if CONFORMAL_Q is not None or DISPATCH_CUTOFF is not None:
            raw = [
                max(decision.distribution.get(opt, 0.0), 1e-12)
                for opt in _ROUTING_OPTIONS
            ]
            scores = [
                math.log(p) / TEMPERATURE + (CLASS_BIAS[i] if CLASS_BIAS else 0.0)
                for i, p in enumerate(raw)
            ]
            peak = max(scores)
            exps = [math.exp(s - peak) for s in scores]
            total = sum(exps)
            probs = [e / total for e in exps]
            index = max(range(len(probs)), key=probs.__getitem__)
            confidence = probs[index]
        abstain_reason = None
        if CONFORMAL_Q is not None:
            included = [i for i, p in enumerate(probs) if p >= 1.0 - CONFORMAL_Q]
            if len(included) == 1:
                index = included[0]
                confidence = probs[index]
            else:
                abstain_reason = (
                    "Conformal prediction set holds {{0}} labels (needs exactly "
                    "1 to dispatch at alpha={{1}})".format(
                        len(included), CALIBRATION.get("alpha")
                    )
                )
        elif DISPATCH_CUTOFF is not None:
            if confidence < DISPATCH_CUTOFF:
                abstain_reason = (
                    "Corrected confidence {{0:.3f}} below the fitted dispatch "
                    "cutoff {{1:.3f}} (target accuracy {{2}})".format(
                        confidence, DISPATCH_CUTOFF, POLICY.get("target_accuracy")
                    )
                )
        else:
            if decision.confidence < CONFIDENCE_THRESHOLD:
                abstain_reason = (
                    "Routing confidence {{0:.3f}} below threshold {{1:.2f}}".format(
                        decision.confidence, CONFIDENCE_THRESHOLD
                    )
                )
        if abstain_reason is None:
            action_key = _ACTION_KEYS[index]
            result = _HANDLERS[action_key](dict(record))
            return {{
                "action": action_key,
                "confidence": confidence,
                "result": result,
            }}

        raise FallbackException(
            abstain_reason + "; falling back.",
            confidence=confidence,
            threshold=DISPATCH_CUTOFF or CONFIDENCE_THRESHOLD,
        )


if __name__ == "__main__":
    import sys

    pipeline = CompiledPipeline()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            print(json.dumps(pipeline.process(json.loads(line)), default=str))
        except FallbackException as exc:
            print(json.dumps({{"action": None, "error": str(exc)}}))
'''
