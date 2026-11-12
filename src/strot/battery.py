"""Battery distribution: a compiled catalog classifier as data, not code.

A compiled battery normally ships as a self-contained ``.py`` with every datum
baked in as a literal (see :mod:`strot.compiler`). That artifact is a complete
application: synthesized ``execute_*`` handlers, a projection function, and the
decision runtime, all inlined. This module adds a *parallel* representation
that strips the asset down to its shippable-as-data core — the decision path
only (record context -> calibrated label + confidence + abstention) — so a
battery can live on a hub as weights + config while the runtime that loads it
stays in the installed package.

A **bundle** is a directory holding two files:

* ``battery.json`` — the task metadata, the routing labels, the fitted
  calibration (temperature / conformal quantile / class bias / dispatch cutoff
  / policy), the optional OOD guard, and a schema fingerprint. Pure JSON, no
  code, torch-free to read.
* ``weights.safetensors`` — the numeric weights the decision runtime needs: the
  distilled logistic head (``HEAD_WEIGHTS`` / ``HEAD_BIAS``) for a distilled
  battery, or the adapter tensors (``ADAPTER`` members + the serving config) for
  a fine-tuned battery. Serialized with ``safetensors``, so it is only touched
  on the torch-backed decision path.

The scope is deliberately narrow. A battery here is a **data-only classifier**:
given a record's projected context string it emits a label, a calibrated
confidence, the full class distribution, and whether it abstained. The
synthesized ``execute_*`` action handlers are app-side glue, not shippable data,
so they are out of scope — :class:`Battery` reproduces the *decision*, never
handler dispatch. Projection (the compiled ``route_context``) is likewise code,
so :meth:`Battery.decide` takes the already-projected context the decision
model sees, exactly as the baked ``CompiledPipeline`` feeds its router.

The runtime is built *on top of* :mod:`strot.router`'s strategies: the saved
adapter tensors are handed straight to :class:`strot.router.FineTunedAdapterStrategy`,
and the distilled head rides the frozen encoder the same strategy family loads.
The calibration + abstention arithmetic is lifted verbatim from the compiled
asset template (:mod:`strot.templates`) so a bundle's decision matches the baked
``.py``'s ``CompiledPipeline`` within floating-point tolerance.

Export captures the data by parsing the compiled asset's own source. Every
datum the decision path needs is already a module-level literal in that source
(``ROUTING_LABELS``, ``TEMPERATURE``, ``CONFORMAL_Q``, ``CLASS_BIAS``,
``DISPATCH_CUTOFF``, ``CALIBRATION``, ``POLICY``, ``OOD``, ``HEAD_WEIGHTS`` /
``HEAD_BIAS`` or ``ADAPTER`` / ``ADAPTER_CONFIG``), so a literal read is both the
single source of truth and a guarantee the bundle cannot drift from the baked
artifact.

Distribution has two sources. A local filesystem bundle (or catalog directory)
is read straight off disk. A **bare battery name** resolves from the hub: a
single pinned Hugging Face *dataset* repo (:data:`DEFAULT_BATTERY_REPO`) holding
a root ``manifest.json`` catalog plus per-battery ``<name>/battery.json`` +
``<name>/weights.safetensors``. :func:`load` / :func:`list_batteries` /
:func:`battery_info` pick disk vs hub by the argument: an existing local
directory stays local, everything else resolves against the hub via
``huggingface_hub`` (imported lazily, needed only on that path; the catalog /
metadata reads stay torch-free).
"""

from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from . import __version__
from .exceptions import RouterConfigurationError

#: Bundle layout version. Bumped only on a breaking change to the on-disk shape
#: so a loader can refuse a future format instead of misreading it.
BUNDLE_FORMAT_VERSION = 1

#: Canonical file names inside a bundle directory.
METADATA_FILENAME = "battery.json"
WEIGHTS_FILENAME = "weights.safetensors"

#: Runtime kinds whose decision path a data-only bundle can reproduce. The
#: zero-shot local (Julia) and hosted runtimes are excluded: their "weights"
#: are an external checkpoint / an API, not shippable bundle data.
SUPPORTED_RUNTIME_KINDS = ("distilled", "fine_tuned")

#: The pinned hub repo a bare battery name resolves against: a single HF
#: *dataset* repo holding every published bundle plus a root ``manifest.json``.
#: Fixed by default so a bare name never resolves somewhere untrusted; a caller
#: can retarget it with ``repo=`` or the ``STROT_BATTERY_REPO`` env var (the arg
#: wins over the env, the env over this default), but it is never inferred from
#: the name itself.
DEFAULT_BATTERY_REPO = "strot-ai/batteries"

#: The catalog file at the hub repo root: ``{"format_version", "batteries":
#: [{name, task, nlu_model, runtime_kind, strot_version, ...}, ...]}``.
MANIFEST_FILENAME = "manifest.json"


# --------------------------------------------------------------------------- #
# Source-literal capture                                                      #
# --------------------------------------------------------------------------- #

#: Module-level literals a compiled asset exposes; read back with
#: ``ast.literal_eval`` (never ``exec``, so capturing a battery never imports
#: torch or runs synthesized code).
_DECISION_LITERALS = (
    "NLU_MODEL",
    "TEMPERATURE",
    "CONFORMAL_Q",
    "CALIBRATION",
    "CLASS_BIAS",
    "DISPATCH_CUTOFF",
    "POLICY",
    "OOD",
    "CONFIDENCE_THRESHOLD",
    "ROUTING_LABELS",
    "DATASET_SCHEMA",
    "HEAD_WEIGHTS",
    "HEAD_BIAS",
    "ADAPTER",
    "ADAPTER_CONFIG",
)


def _read_literals(source: str) -> Dict[str, Any]:
    """Pull every decision-path literal out of a compiled asset's source.

    Parses (never executes) the module and ``literal_eval``s each top-level
    assignment we recognize. Absent names simply do not appear in the result —
    a distilled asset has no ``ADAPTER``, a fine-tuned one has no
    ``HEAD_WEIGHTS``.
    """
    tree = ast.parse(source)
    found: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if target.id in _DECISION_LITERALS and node.value is not None:
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass
    return found


def _schema_fingerprint(schema: Mapping[str, Any]) -> str:
    """A stable short digest of the dataset schema.

    Lets a loader warn when a bundle is applied to records whose structure no
    longer matches what it was compiled against, without pinning exact sample
    values. Pure stdlib; deterministic across runs.
    """
    import hashlib

    fields = [
        (f.get("name"), f.get("semantic_type"), f.get("python_type"))
        for f in schema.get("fields", [])
    ]
    canonical = json.dumps(sorted(map(list, fields)), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Calibration + abstention: lifted from the compiled asset template            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Calibration:
    """The fitted correction + abstention knobs a bundle carries.

    Mirrors the ``TEMPERATURE`` / ``CONFORMAL_Q`` / ``CLASS_BIAS`` /
    ``DISPATCH_CUTOFF`` / ``CALIBRATION`` / ``POLICY`` literals of a compiled
    asset one-to-one; the arithmetic in :func:`apply_decision` is the template's
    ``CompiledPipeline.process`` correction block, verbatim.
    """

    temperature: float = 1.0
    conformal_q: Optional[float] = None
    class_bias: Optional[List[float]] = None
    dispatch_cutoff: Optional[float] = None
    confidence_threshold: float = 0.40
    calibration: Dict[str, Any] = None  # type: ignore[assignment]
    policy: Dict[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "conformal_q": self.conformal_q,
            "class_bias": self.class_bias,
            "dispatch_cutoff": self.dispatch_cutoff,
            "confidence_threshold": self.confidence_threshold,
            "calibration": self.calibration or {},
            "policy": self.policy or {},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Calibration":
        return cls(
            temperature=float(data.get("temperature", 1.0)),
            conformal_q=data.get("conformal_q"),
            class_bias=data.get("class_bias"),
            dispatch_cutoff=data.get("dispatch_cutoff"),
            confidence_threshold=float(data.get("confidence_threshold", 0.40)),
            calibration=dict(data.get("calibration") or {}),
            policy=dict(data.get("policy") or {}),
        )


@dataclass(frozen=True)
class Decision:
    """A battery's verdict on one record.

    ``abstained`` is ``True`` when the fitted conformal / dispatch-cutoff gate
    refused to commit (the compiled asset raises ``FallbackException`` here);
    ``label`` / ``confidence`` then report the leading candidate that was
    withheld, and ``abstain_reason`` explains why.
    """

    label: str
    confidence: float
    probabilities: Dict[str, float]
    abstained: bool
    abstain_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
        }


def apply_decision(
    distribution: Mapping[str, float],
    labels: Sequence[str],
    calibration: Calibration,
) -> Decision:
    """Turn a raw class distribution into a calibrated, abstention-aware verdict.

    This is the ``CompiledPipeline.process`` correction + abstention block
    (:mod:`strot.templates`) lifted out of the baked asset, operating on the
    router's raw per-label distribution instead of dispatching a handler:

    * With no conformal quantile and no dispatch cutoff fitted, the raw argmax
      and its probability pass through unchanged (never hard-abstain).
    * Otherwise temperature + per-class bias correct the distribution, and the
      verdict either survives the conformal singleton test / the dispatch cutoff
      or is marked abstained — matching the artifact's ``FallbackException``.
    """
    options = list(labels)
    raw = [max(distribution.get(opt, 0.0), 1e-12) for opt in options]
    index = max(range(len(raw)), key=raw.__getitem__)
    confidence = raw[index]
    probs = list(raw)

    conformal_q = calibration.conformal_q
    dispatch_cutoff = calibration.dispatch_cutoff
    class_bias = calibration.class_bias
    temperature = calibration.temperature

    if conformal_q is not None or dispatch_cutoff is not None:
        scores = [
            math.log(p) / temperature + (class_bias[i] if class_bias else 0.0)
            for i, p in enumerate(raw)
        ]
        peak = max(scores)
        exps = [math.exp(s - peak) for s in scores]
        total = sum(exps)
        probs = [e / total for e in exps]
        index = max(range(len(probs)), key=probs.__getitem__)
        confidence = probs[index]

    abstain_reason: Optional[str] = None
    if conformal_q is not None:
        included = [i for i, p in enumerate(probs) if p >= 1.0 - conformal_q]
        if len(included) == 1:
            index = included[0]
            confidence = probs[index]
        else:
            abstain_reason = (
                "Conformal prediction set holds {0} labels (needs exactly "
                "1 to dispatch at alpha={1})".format(
                    len(included), (calibration.calibration or {}).get("alpha")
                )
            )
    elif dispatch_cutoff is not None:
        if confidence < dispatch_cutoff:
            abstain_reason = (
                "Corrected confidence {0:.3f} below the fitted dispatch "
                "cutoff {1:.3f} (target accuracy {2})".format(
                    confidence, dispatch_cutoff,
                    (calibration.policy or {}).get("target_accuracy"),
                )
            )

    return Decision(
        label=options[index],
        confidence=confidence,
        probabilities=dict(zip(options, probs)),
        abstained=abstain_reason is not None,
        abstain_reason=abstain_reason,
    )


# --------------------------------------------------------------------------- #
# Export                                                                       #
# --------------------------------------------------------------------------- #


def _resolve_source(asset: Any) -> str:
    """Accept a CompiledAsset, a source string, or a .py path and return source."""
    source = getattr(asset, "source", None)
    if isinstance(source, str):
        return source
    if isinstance(asset, (str, Path)):
        candidate = Path(asset)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
        return str(asset)
    raise RouterConfigurationError(
        "export_bundle expects a CompiledAsset, a compiled-asset source string, "
        f"or a path to a compiled .py; got {type(asset).__name__!r}."
    )


def _save_weights(path: Path, tensors: Mapping[str, Any], metadata: Mapping[str, str]) -> None:
    """Write numeric weights as safetensors (torch-free, via the numpy API).

    safetensors needs rectangular arrays. The distilled head and the adapter
    tensors are already rectangular python lists; each is flattened with its
    shape recorded in the tensor key so the loader can restore it without a
    separate manifest. A missing ``safetensors`` install (part of the [local] /
    [finetune] extras) raises the same actionable error the runtime uses.
    """
    try:
        import numpy as np
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover - env guard
        raise RouterConfigurationError(
            "Writing a battery's weights.safetensors needs safetensors + numpy: "
            "pip install strot-ai[local]"
        ) from exc

    arrays: Dict[str, Any] = {}
    for key, value in tensors.items():
        arrays[key] = np.asarray(value, dtype=np.float32)
    save_file(arrays, str(path), metadata=dict(metadata))


def _collect_bundle(
    asset: Any,
    *,
    name: Optional[str] = None,
    question: Optional[str] = None,
) -> "tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]":
    """Capture a compiled asset's decision data: ``(metadata, tensors, meta)``.

    Torch-free: reads module-level literals out of the asset source (never
    ``exec``), builds the ``battery.json`` payload, and collects the numeric
    weights as plain python lists. The actual safetensors write is deferred to
    :func:`export_bundle`, so the full ``battery.json`` content is assertable
    without safetensors / torch installed.
    """
    source = _resolve_source(asset)
    literals = _read_literals(source)

    runtime_kind = getattr(asset, "runtime_kind", None)
    if runtime_kind not in SUPPORTED_RUNTIME_KINDS:
        # Infer from the captured literals when the caller passed raw source.
        if "ADAPTER" in literals:
            runtime_kind = "fine_tuned"
        elif "HEAD_WEIGHTS" in literals:
            runtime_kind = "distilled"
    if runtime_kind not in SUPPORTED_RUNTIME_KINDS:
        raise RouterConfigurationError(
            "Only distilled / fine_tuned batteries carry shippable weights; got "
            f"runtime_kind={runtime_kind!r}. (A zero-shot local or hosted asset "
            "has no bundle data — distribute it as the compiled .py.)"
        )

    schema = literals.get("DATASET_SCHEMA", {}) or {}
    labels = literals.get("ROUTING_LABELS", {}) or {}
    calibration = Calibration(
        temperature=float(literals.get("TEMPERATURE", 1.0) or 1.0),
        conformal_q=literals.get("CONFORMAL_Q"),
        class_bias=literals.get("CLASS_BIAS"),
        dispatch_cutoff=literals.get("DISPATCH_CUTOFF"),
        confidence_threshold=float(literals.get("CONFIDENCE_THRESHOLD", 0.40)),
        calibration=literals.get("CALIBRATION") or {},
        policy=literals.get("POLICY") or {},
    )

    goal = getattr(asset, "goal", None)
    battery_name = name or (goal or "battery")
    nlu_model = literals.get("NLU_MODEL") or getattr(asset, "nlu_model", None)

    metadata: Dict[str, Any] = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "strot_version": __version__,
        "name": battery_name,
        "task": {"goal": goal, "question": question},
        "question": question,
        "nlu_model": nlu_model,
        "runtime_kind": runtime_kind,
        "labels": labels,
        "calibration": calibration.to_dict(),
        "ood": literals.get("OOD") or None,
        "schema_fingerprint": _schema_fingerprint(schema),
        "schema": schema,
    }

    # Gather the numeric weights for this runtime, plus the sidecar config that
    # describes their shape / architecture (kept in battery.json since it is
    # small structured metadata, not a tensor).
    tensors: Dict[str, Any] = {}
    weights_metadata: Dict[str, str] = {"runtime_kind": runtime_kind}
    if runtime_kind == "distilled":
        tensors["head_weights"] = literals["HEAD_WEIGHTS"]
        tensors["head_bias"] = literals["HEAD_BIAS"]
    else:  # fine_tuned
        adapters = literals["ADAPTER"]
        adapters = adapters if isinstance(adapters, list) else [adapters]
        metadata["adapter_config"] = literals.get("ADAPTER_CONFIG") or {}
        metadata["n_adapters"] = len(adapters)
        # Adapter tensors are keyed per-member, per-parameter (member index +
        # the original parameter name). The parameter-name list per member is
        # recorded in battery.json so the loader rebuilds each adapter faithfully.
        adapter_param_names: List[List[str]] = []
        for member_ix, adapter in enumerate(adapters):
            names: List[str] = []
            for pname, value in adapter.items():
                tensors[f"adapter.{member_ix}.{pname}"] = value
                names.append(pname)
            adapter_param_names.append(names)
        metadata["adapter_param_names"] = adapter_param_names

    return metadata, tensors, weights_metadata


def export_bundle(
    asset: Any,
    directory: Union[str, Path],
    *,
    name: Optional[str] = None,
    question: Optional[str] = None,
) -> Path:
    """Write a data-only bundle (``battery.json`` + ``weights.safetensors``).

    ``asset`` is a :class:`strot.compiler.CompiledAsset` (preferred — its
    ``source`` and metadata are read directly), or the compiled source string /
    a path to a compiled ``.py``. The decision-path data is captured by reading
    the asset's module-level literals, so the bundle cannot drift from the baked
    artifact. The existing baked-``.py`` path is untouched: this only *reads* the
    asset.

    Only ``distilled`` / ``fine_tuned`` runtimes are exportable as data — the
    zero-shot local and hosted runtimes carry no shippable weights.
    """
    metadata, tensors, weights_metadata = _collect_bundle(
        asset, name=name, question=question
    )
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    (target / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    _save_weights(target / WEIGHTS_FILENAME, tensors, weights_metadata)
    return target


# --------------------------------------------------------------------------- #
# Data-parameterized runtime                                                  #
# --------------------------------------------------------------------------- #


class Battery:
    """A data-only catalog classifier loaded from a bundle.

    Wraps a :mod:`strot.router` strategy fed the bundle's own weights and layers
    the lifted calibration + abstention math on top, so ``decide`` reproduces a
    compiled asset's ``CompiledPipeline`` decision without any synthesized code.

    Build with :meth:`from_bundle`. The torch-backed base model loads lazily on
    the first :meth:`decide`, via the same checkpoint path the runtime uses
    (``STROT_NLU_CHECKPOINT`` / :func:`strot.prefetch_nlu`); a missing extra /
    checkpoint surfaces an actionable error, never a cryptic import failure.
    """

    def __init__(
        self,
        runtime_kind: str,
        labels: Mapping[str, str],
        calibration: Calibration,
        *,
        nlu_model: Optional[str] = None,
        head_weights: Optional[List[List[float]]] = None,
        head_bias: Optional[List[float]] = None,
        adapters: Optional[List[Dict[str, Any]]] = None,
        adapter_config: Optional[Dict[str, Any]] = None,
        ood: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if runtime_kind not in SUPPORTED_RUNTIME_KINDS:
            raise RouterConfigurationError(
                f"Battery runtime_kind must be one of {SUPPORTED_RUNTIME_KINDS}; "
                f"got {runtime_kind!r}."
            )
        self.runtime_kind = runtime_kind
        #: Ordered action keys == routing option indices (dict order is the
        #: compiled registry order, matching the baked asset).
        self.action_keys: List[str] = list(labels)
        #: The short routing labels the decision model actually scored; option i
        #: is ``labels[action_keys[i]]``.
        self.labels: List[str] = [labels[key] for key in self.action_keys]
        self.calibration = calibration
        self.nlu_model = nlu_model
        self.ood = ood or None
        self.metadata = metadata or {}
        self._head_weights = head_weights
        self._head_bias = head_bias
        self._adapters = adapters
        self._adapter_config = adapter_config
        self._strategy: Any = None
        self._embed: Any = None

    # -- loading ---------------------------------------------------------- #

    @classmethod
    def from_bundle(cls, path: Union[str, Path]) -> "Battery":
        """Load a bundle directory into a ready (lazily-initialized) battery."""
        directory = Path(path)
        meta_path = directory / METADATA_FILENAME
        if not meta_path.exists():
            raise RouterConfigurationError(
                f"No {METADATA_FILENAME} in {directory} — not a battery bundle."
            )
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        fmt = metadata.get("format_version")
        if fmt != BUNDLE_FORMAT_VERSION:
            raise RouterConfigurationError(
                f"Bundle format_version {fmt} is not supported by this strot "
                f"(expects {BUNDLE_FORMAT_VERSION}); upgrade strot-ai."
            )
        runtime_kind = metadata.get("runtime_kind")
        labels = metadata.get("labels", {})
        calibration = Calibration.from_dict(metadata.get("calibration", {}))

        head_weights = head_bias = None
        adapters = adapter_config = None
        if runtime_kind == "distilled":
            head_weights, head_bias = _load_head_weights(directory)
        elif runtime_kind == "fine_tuned":
            adapters = _load_adapter_weights(directory, metadata)
            adapter_config = metadata.get("adapter_config", {})

        return cls(
            runtime_kind=runtime_kind,
            labels=labels,
            calibration=calibration,
            nlu_model=metadata.get("nlu_model"),
            head_weights=head_weights,
            head_bias=head_bias,
            adapters=adapters,
            adapter_config=adapter_config,
            ood=metadata.get("ood"),
            metadata=metadata,
        )

    def _ensure_runtime(self) -> None:
        """Lazily build the torch-backed decision runtime from the bundle data.

        Reuses :mod:`strot.router`: a fine-tuned battery hands its adapters +
        config to :class:`strot.router.FineTunedAdapterStrategy`; a distilled
        battery loads the frozen encoder embedder (:func:`strot.distill.
        make_encoder_embedder`, the same encoder that family serves with) and
        runs the saved logistic head. Both resolve the base checkpoint through
        the shared runtime path.
        """
        if self.runtime_kind == "fine_tuned":
            if self._strategy is not None:
                return
            from .router import FineTunedAdapterStrategy

            self._strategy = FineTunedAdapterStrategy(
                adapters=self._adapters,
                config=self._adapter_config or {},
                nlu_model=self.nlu_model or "",
            )
        else:  # distilled
            if self._embed is not None:
                return
            from . import distill as distill_mod

            checkpoint = _resolve_checkpoint(self.nlu_model)
            self._embed = distill_mod.make_encoder_embedder(checkpoint)

    # -- inference -------------------------------------------------------- #

    def distribution(self, context: str) -> Dict[str, float]:
        """Raw per-label distribution the decision runtime assigns to ``context``.

        ``context`` is the already-projected string the compiled asset's router
        sees (the output of the synthesized ``route_context``); projection is
        app-side code, not bundle data, so the caller supplies it.
        """
        self._ensure_runtime()
        if self.runtime_kind == "fine_tuned":
            decision = self._strategy.select(context, "", self.labels)
            return dict(decision.distribution)
        # distilled: frozen-encoder embedding + the saved logistic head.
        from .distill import head_predict

        vector = self._embed([context])[0]
        probs = head_predict(vector, self._head_weights, self._head_bias)
        return dict(zip(self.labels, probs))

    def is_out_of_distribution(self, context: str) -> bool:
        """True iff the OOD guard is fitted and ``context`` is out-of-distribution.

        Mirrors the compiled asset's drift gate: embed with the frozen encoder
        (only the distilled runtime embeds) and compare to the fitted per-class
        centroids. Fine-tuned batteries and un-fitted OOD always return False.
        """
        if not self.ood or self.runtime_kind != "distilled":
            return False
        from .router import is_ood

        self._ensure_runtime()
        vector = self._embed([context])[0]
        return is_ood(vector, self.ood["centroids"], self.ood["threshold"])

    def decide(self, context: str) -> Dict[str, Any]:
        """Classify one projected context into a calibrated, gated verdict.

        Returns ``{label, confidence, probabilities, abstained, abstain_reason}``
        — the data-only analogue of ``CompiledPipeline.process``'s decision
        (sans handler dispatch). An out-of-distribution record abstains up front,
        exactly as the baked asset refuses it before routing.
        """
        if self.is_out_of_distribution(context):
            # The OOD label is the leading candidate for reporting only; the
            # verdict is an abstention, matching the asset's "ood" FallbackException.
            raw = self.distribution(context)
            top = max(raw, key=raw.__getitem__)
            return Decision(
                label=top,
                confidence=0.0,
                probabilities=raw,
                abstained=True,
                abstain_reason=(
                    "Record is out-of-distribution (farther than the fitted "
                    "threshold from every class centroid)."
                ),
            ).to_dict()
        distribution = self.distribution(context)
        return apply_decision(distribution, self.labels, self.calibration).to_dict()


# --------------------------------------------------------------------------- #
# Weight (de)serialization helpers                                            #
# --------------------------------------------------------------------------- #


def _load_tensors(directory: Path) -> Dict[str, Any]:
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover - env guard
        raise RouterConfigurationError(
            "Loading a battery's weights.safetensors needs safetensors + numpy: "
            "pip install strot-ai[local]"
        ) from exc
    weights_path = directory / WEIGHTS_FILENAME
    if not weights_path.exists():
        raise RouterConfigurationError(
            f"No {WEIGHTS_FILENAME} in {directory}; the bundle is incomplete."
        )
    return load_file(str(weights_path))


def _load_head_weights(directory: Path) -> "tuple[List[List[float]], List[float]]":
    tensors = _load_tensors(directory)
    head_weights = tensors["head_weights"].tolist()
    head_bias = tensors["head_bias"].tolist()
    return head_weights, head_bias


def _load_adapter_weights(
    directory: Path, metadata: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    tensors = _load_tensors(directory)
    param_names: List[List[str]] = metadata.get("adapter_param_names", [])
    adapters: List[Dict[str, Any]] = []
    for member_ix, names in enumerate(param_names):
        adapter: Dict[str, Any] = {}
        for pname in names:
            adapter[pname] = tensors[f"adapter.{member_ix}.{pname}"].tolist()
        adapters.append(adapter)
    return adapters


def _resolve_checkpoint(nlu_model: Optional[str]) -> str:
    """Resolve the base checkpoint the way the runtime does, with a clear error.

    Env override > a local checkpoint dir > HF download (lazy import). A missing
    checkpoint and no [local] extra raises an actionable message pointing at the
    extra / :func:`strot.prefetch_nlu`, not a cryptic import error deep in the
    encoder loader.
    """
    import os

    override = os.environ.get("STROT_NLU_CHECKPOINT")
    if override:
        return override
    if nlu_model and Path(nlu_model).is_dir():
        return nlu_model
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RouterConfigurationError(
            "This battery's decision runtime needs the local model checkpoint. "
            "Set STROT_NLU_CHECKPOINT to a prefetched checkpoint directory, or "
            "install the extra and prefetch:\n"
            "  pip install strot-ai[local]\n"
            "  python -c \"import strot; strot.prefetch_nlu()\""
        ) from exc
    if not nlu_model:
        raise RouterConfigurationError(
            "This battery has no nlu_model recorded and STROT_NLU_CHECKPOINT is "
            "unset; cannot resolve the base checkpoint. Set STROT_NLU_CHECKPOINT."
        )
    return snapshot_download(nlu_model)


# --------------------------------------------------------------------------- #
# Hub resolution (a bare battery name -> files cached from the HF hub)         #
# --------------------------------------------------------------------------- #


def _resolve_repo(repo: Optional[str]) -> str:
    """Pick the hub repo to resolve a bare name against.

    Precedence: explicit ``repo=`` arg > ``STROT_BATTERY_REPO`` env var >
    :data:`DEFAULT_BATTERY_REPO`. The default stays pinned so a bare name never
    silently resolves somewhere untrusted; overriding it is an explicit act.
    """
    import os

    if repo is not None:
        return repo
    return os.environ.get("STROT_BATTERY_REPO") or DEFAULT_BATTERY_REPO


def _hf_hub_download(repo: str, filename: str, revision: Optional[str]) -> str:
    """Download one file from the hub dataset repo, returning its cached path.

    ``huggingface_hub`` is only needed on this hub path, so it is imported
    lazily here (never at module import) and a missing install surfaces the same
    actionable ``strot-ai[local]`` guidance the runtime uses. Reading a battery
    catalog / metadata needs only this dependency — no torch — so the import is
    deliberately kept independent of the runtime's safetensors/torch guards.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RouterConfigurationError(
            "Resolving a battery by name downloads it from the Hugging Face hub, "
            "which needs huggingface_hub. Install the extra or the package "
            "directly:\n"
            "  pip install strot-ai[local]\n"
            "  pip install huggingface_hub\n"
            "(Or pass a local bundle directory path instead of a name.)"
        ) from exc
    return hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=filename,
        revision=revision,
    )


def _hub_file_missing(exc: Exception) -> bool:
    """True iff a hub download failed because the requested file is not there.

    A 404 (``EntryNotFoundError`` / ``RepositoryNotFoundError``) means an
    unknown battery; any other failure (offline, auth, transport) is a different
    class of problem and should surface its own message rather than be reported
    as "not found".
    """
    name = type(exc).__name__
    if name in ("EntryNotFoundError", "RepositoryNotFoundError"):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 404)


def _download_battery_json(repo: str, name: str, revision: Optional[str]) -> Dict[str, Any]:
    """Fetch and parse ``<name>/battery.json`` from the hub, torch-free.

    A missing battery is turned into an actionable "not found" error pointing
    the caller at :func:`list_batteries`; other download failures propagate.
    """
    try:
        path = _hf_hub_download(repo, f"{name}/{METADATA_FILENAME}", revision)
    except RouterConfigurationError:
        raise
    except Exception as exc:
        if _hub_file_missing(exc):
            raise RouterConfigurationError(
                f"battery {name!r} not found in {repo!r}; run "
                "strot.list_batteries() to see available batteries."
            ) from exc
        raise RouterConfigurationError(
            f"Could not download battery {name!r} from {repo!r}: {exc}. "
            "Check your network connection and the battery name "
            "(strot.list_batteries() lists what is published)."
        ) from exc
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_bundle_from_hub(name: str, repo: str, revision: Optional[str]) -> Path:
    """Download a battery's two files from the hub, returning their directory.

    ``hf_hub_download`` caches each file under the HF cache and lays the repo's
    ``<name>/battery.json`` + ``<name>/weights.safetensors`` out under a common
    snapshot directory, so the two cached paths share a parent that is a valid
    bundle directory :meth:`Battery.from_bundle` can read unchanged.
    """
    # Fetch battery.json first so an unknown name fails fast with the "not
    # found" guidance before we attempt the (larger) weights file.
    try:
        meta_path = _hf_hub_download(repo, f"{name}/{METADATA_FILENAME}", revision)
    except RouterConfigurationError:
        raise
    except Exception as exc:
        if _hub_file_missing(exc):
            raise RouterConfigurationError(
                f"battery {name!r} not found in {repo!r}; run "
                "strot.list_batteries() to see available batteries."
            ) from exc
        raise RouterConfigurationError(
            f"Could not download battery {name!r} from {repo!r}: {exc}. "
            "Check your network connection and the battery name."
        ) from exc
    try:
        _hf_hub_download(repo, f"{name}/{WEIGHTS_FILENAME}", revision)
    except RouterConfigurationError:
        raise
    except Exception as exc:
        raise RouterConfigurationError(
            f"Downloaded {name!r}'s metadata but not its weights from {repo!r}: "
            f"{exc}. The published bundle may be incomplete."
        ) from exc
    return Path(meta_path).parent


def _looks_like_path(name_or_path: Union[str, Path]) -> bool:
    """True iff the argument names an existing local bundle directory.

    Name-vs-path disambiguation is deliberately conservative: only an existing
    local directory takes the Stage 1 local path; everything else (a bare name,
    or a path that is not a directory on disk) resolves against the hub. A hub
    battery name is a plain identifier, never a filesystem path, so there is no
    ambiguity for a published catalog.
    """
    return Path(name_or_path).is_dir()


# --------------------------------------------------------------------------- #
# Public catalog helpers (local filesystem + hub)                             #
# --------------------------------------------------------------------------- #


def load(
    name_or_path: Union[str, Path],
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
) -> Battery:
    """Load a battery into a :class:`Battery`, from disk or the hub.

    ``name_or_path`` is either an existing local bundle directory (loaded
    unchanged, Stage 1 behavior) or a bare battery name resolved from the hub.
    A name downloads ``<name>/battery.json`` + ``<name>/weights.safetensors``
    from the pinned ``repo`` dataset (``huggingface_hub`` caches them), then
    builds the battery from the cached files. ``repo`` / ``STROT_BATTERY_REPO``
    retarget the repo; ``revision`` pins a branch/tag/commit.
    """
    if _looks_like_path(name_or_path):
        return Battery.from_bundle(name_or_path)
    directory = _resolve_bundle_from_hub(
        str(name_or_path), _resolve_repo(repo), revision
    )
    return Battery.from_bundle(directory)


def battery_info(
    name_or_path: Union[str, Path],
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a battery's ``battery.json`` without loading any weights.

    ``name_or_path`` is a local bundle directory (or the path to its
    ``battery.json``) read from disk, or a bare battery name whose
    ``<name>/battery.json`` is downloaded from the hub. Torch-free either way:
    this only reads metadata, so it needs ``huggingface_hub`` but never the
    runtime (torch/safetensors).
    """
    candidate = Path(name_or_path)
    # A direct path to a bundle's battery.json, or a bundle directory on disk,
    # reads from local filesystem (Stage 1); anything else is a hub name.
    if candidate.name == METADATA_FILENAME and candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    if candidate.is_dir():
        meta_path = candidate / METADATA_FILENAME
        if not meta_path.exists():
            raise RouterConfigurationError(
                f"No {METADATA_FILENAME} under {name_or_path!r}; not a battery bundle."
            )
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return _download_battery_json(_resolve_repo(repo), str(name_or_path), revision)


def list_batteries(
    source: Optional[Union[str, Path]] = None,
    *,
    repo: Optional[str] = None,
    revision: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List published battery metadata, from a local catalog dir or the hub.

    With ``source`` a local directory, each entry is a bundle's ``battery.json``
    plus a ``path`` key, for every bundle directly under it (Stage 1 behavior;
    directories without a ``battery.json`` are skipped). With ``source`` omitted
    (or not a local directory), the hub repo's root ``manifest.json`` is
    downloaded and its ``batteries`` list returned — one summary row per
    published battery. Torch-free either way.
    """
    if source is not None and Path(source).is_dir():
        root = Path(source)
        out: List[Dict[str, Any]] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            meta_path = child / METADATA_FILENAME
            if not meta_path.exists():
                continue
            info = json.loads(meta_path.read_text(encoding="utf-8"))
            info["path"] = str(child)
            out.append(info)
        return out

    resolved_repo = _resolve_repo(repo)
    try:
        path = _hf_hub_download(resolved_repo, MANIFEST_FILENAME, revision)
    except RouterConfigurationError:
        raise
    except Exception as exc:
        raise RouterConfigurationError(
            f"Could not download the battery catalog ({MANIFEST_FILENAME}) from "
            f"{resolved_repo!r}: {exc}. Check your network connection; pass a "
            "local catalog directory as the first argument to list on-disk "
            "bundles instead."
        ) from exc
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(manifest.get("batteries", []))
