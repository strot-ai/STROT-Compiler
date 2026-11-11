"""PII masking at the teacher boundary.

The cloud LLM is confined to compile time and never sees production traffic,
but at compile time one boundary does hand real sample records to a cloud
teacher - the labeling call that turns records into supervision. This module
masks at that boundary, so the records reaching the teacher are scrubbed of
obvious PII first.

:func:`mask_records` applies a list of pluggable masker callables (each a
``Callable[[str], str]``) to every record. The default list (see
:func:`default_maskers`) is a thin regex pre-pass for format-defined tokens
(emails, SSN, Luhn card, secrets, IBAN, IP/MAC/phone) followed by GLiNER, the
default masker for linguistic PII (names, addresses, orgs, locations). GLiNER
needs the ``[pii]`` extra; without it the default degrades to regex-only with a
warning. None of this is a compliance guarantee: deployments can pass their own
maskers (a stronger NER/PII model, an allowlist projection in ``route_context``,
etc.). Masking runs over the string projection of each field value; structure
(dict keys, non-string values) is preserved so downstream serialization is
unchanged.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .exceptions import RouterConfigurationError

#: A masker maps a string to a redacted string.
Masker = Callable[[str], str]

# Default regex redactors (baseline, not a guarantee): structured-identifier
# patterns regex can catch with low false positives. Unstructured PII (names,
# addresses) still needs an NER masker plugged in via the `maskers` argument -
# that is the documented upgrade path.

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
#: SSN-like: 3-2-4 digit groups separated by - or space.
_SSN_RE = re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")
#: Credit-card-like: 13-19 digit runs, Luhn-validated (see mask_credit_card).
_CC_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
#: IPv4 with validated octets (0-255), and a basic IPv6 form.
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
_IPV6_RE = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b")
#: MAC address (6 colon/dash-separated hex pairs).
_MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
#: IBAN: 2 country letters + 2 check digits + 11-30 alphanumerics.
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
#: Common secret/credential shapes: AWS access keys, JWTs, provider keys
#: (sk-/ghp_/xox...), and `Bearer <token>`. Keeps keys out of teacher prompts.
_SECRET_RES = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{12,}\b"),
)
#: Phone-like: optional +cc, then 10+ digits with common separators. Checked
#: LAST so the more specific digit patterns above win their spans first.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?"
    r"\d{3}[\s.\-]?\d{3,4}(?:[\s.\-]?\d{2,4})?(?!\d)"
)


def _luhn_ok(run: str) -> bool:
    """True if the digit run passes the Luhn checksum (a real card number)."""
    digits = [int(c) for c in run if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mask_email(text: str) -> str:
    """Redact email addresses."""
    return _EMAIL_RE.sub("[EMAIL]", text)


def mask_ssn(text: str) -> str:
    """Redact SSN-like 3-2-4 digit sequences."""
    return _SSN_RE.sub("[SSN]", text)


def mask_credit_card(text: str) -> str:
    """Redact credit-card-like runs that pass the Luhn checksum.

    Luhn validation keeps the matcher precise: a random 13-19 digit id or a
    long phone number is not redacted unless it is actually a valid card
    number, which avoids over-masking the teacher's inputs.
    """
    return _CC_RE.sub(lambda m: "[CARD]" if _luhn_ok(m.group(0)) else m.group(0), text)


def mask_ip(text: str) -> str:
    """Redact IPv4 (validated octets) and basic IPv6 addresses."""
    return _IPV6_RE.sub("[IP]", _IPV4_RE.sub("[IP]", text))


def mask_mac(text: str) -> str:
    """Redact MAC addresses."""
    return _MAC_RE.sub("[MAC]", text)


def mask_iban(text: str) -> str:
    """Redact IBAN bank account numbers."""
    return _IBAN_RE.sub("[IBAN]", text)


def mask_secret(text: str) -> str:
    """Redact common API-key / token / credential shapes."""
    for pattern in _SECRET_RES:
        text = pattern.sub("[SECRET]", text)
    return text


def mask_phone(text: str) -> str:
    """Redact phone-number-like digit sequences."""
    return _PHONE_RE.sub("[PHONE]", text)


#: A thin precision pre-pass for format-defined tokens only: the shapes regex
#: catches cleanly with near-zero false positives (email, secrets, IBAN, IP,
#: MAC, SSN, Luhn-validated card, phone). It runs before GLiNER so it pulls
#: exact-format tokens out of the text as whole spans; the broad phone matcher
#: is still last so the more specific digit patterns claim their spans first.
#: Linguistic PII (names, addresses, orgs) is GLiNER's job, not regex's.
DEFAULT_MASKERS: List[Masker] = [
    mask_email, mask_secret, mask_iban, mask_mac, mask_ip,
    mask_ssn, mask_credit_card, mask_phone,
]


# GLiNER (zero-shot NER) is the default masker for linguistic PII: person
# names, addresses, organizations, locations - the entities that have no fixed
# shape and that regex cannot catch. The regex pass above is a precision
# pre-pass for format-defined tokens only; GLiNER carries the headline
# coverage. It needs the [pii] extra (torch + the gliner model). When the extra
# is installed GLiNER is on by default; when it is not, default_maskers() warns
# and degrades to regex-only. Compile-time only.

#: Lightweight zero-shot NER checkpoint (medium variant, CPU-friendly).
DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"
#: Unstructured entity types GLiNER covers that regex cannot.
DEFAULT_PII_ENTITIES: List[str] = [
    "person", "person name", "address", "location",
    "organization", "nationality", "date of birth",
]


def gliner_masker(
    entity_types: Optional[Sequence[str]] = None,
    model_id: str = DEFAULT_GLINER_MODEL,
    threshold: float = 0.5,
) -> Masker:
    """Build a NER masker that redacts GLiNER-detected entity spans.

    Returns a ``Masker`` that replaces each detected entity with ``[<TYPE>]``.
    The model is loaded once here and reused for every call. Needs the ``[pii]``
    extra; raises :class:`RouterConfigurationError` otherwise so callers can
    fall back to the regex baseline.
    """
    try:
        from gliner import GLiNER
    except ImportError as exc:  # pragma: no cover - env guard
        raise RouterConfigurationError(
            "NER masking needs the pii extra: pip install strot-ai[pii]"
        ) from exc
    labels = list(entity_types or DEFAULT_PII_ENTITIES)
    model = GLiNER.from_pretrained(model_id)

    def _mask(text: str) -> str:
        entities = model.predict_entities(text, labels, threshold=threshold)
        # Redact right-to-left so earlier spans keep their offsets.
        for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
            tag = "[" + str(ent["label"]).upper().replace(" ", "_") + "]"
            text = text[: ent["start"]] + tag + text[ent["end"]:]
        return text

    return _mask


_NER_MASKER_CACHE: Optional[Masker] = None


def default_maskers() -> List[Masker]:
    """The default masker list for the teacher boundary.

    GLiNER is the default masker. The returned list is ``[<regex pre-pass>,
    <GLiNER masker>]``: the thin regex pre-pass runs first to claim
    format-defined tokens (emails, cards, IPs, secrets) as clean whole spans,
    then GLiNER runs over the remainder for the linguistic PII it owns - names,
    addresses, orgs, locations. Order matters: running NER first can tag part of
    an email domain as an organization and leave a mangled ``jane@[ORG]`` that
    the email regex can no longer match, leaking the local part. Regex-first
    avoids that.

    GLiNER needs the ``[pii]`` extra. When it is importable it is included by
    default, loaded once and cached. If the import fails, this warns and
    degrades to the regex pre-pass alone, so base installs still work.
    Compile-time only.
    """
    global _NER_MASKER_CACHE
    try:
        if _NER_MASKER_CACHE is None:
            _NER_MASKER_CACHE = gliner_masker()
        return [*DEFAULT_MASKERS, _NER_MASKER_CACHE]
    except RouterConfigurationError:
        warnings.warn(
            "GLiNER not installed; PII masking is falling back to regex-only, "
            "which misses names/addresses. Install strot-ai[pii].",
            stacklevel=2,
        )
        return list(DEFAULT_MASKERS)


def mask_text(text: str, maskers: Sequence[Masker] = DEFAULT_MASKERS) -> str:
    """Apply every masker to ``text`` in order."""
    for masker in maskers:
        text = masker(text)
    return text


def mask_records(
    records: Sequence[Mapping[str, Any]],
    maskers: Sequence[Masker] = DEFAULT_MASKERS,
) -> List[Dict[str, Any]]:
    """Return copies of ``records`` with every string field value masked.

    Each masker in ``maskers`` is applied, in order, to the string form of
    every field value. Non-string values are left untouched (nothing to
    redact in a bare int); dict structure and key order are preserved. Pass a
    custom ``maskers`` list to swap the baseline regexes for a stronger PII
    hook - this function only orchestrates; it makes no promises about the
    maskers' recall.
    """
    masked: List[Dict[str, Any]] = []
    for record in records:
        row: Dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, str):
                row[key] = mask_text(value, maskers)
            else:
                row[key] = value
        masked.append(row)
    return masked
