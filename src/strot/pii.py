"""Structured-PII detection for `pii_scan` (regex, no model, no network).

STROT itself never sends records anywhere - it embeds locally - so there is no
"teacher boundary" to mask at. This module is a detection baseline: a zero-dep
regex pass over format-defined tokens (email, SSN, Luhn card, secrets, IBAN,
IP/MAC/phone) that :func:`pii_scan` uses to tell an agent what structured PII a
record carries *before the agent feeds it to its own LLM*. It is an advisory
signal, not a compliance guarantee, and not a privacy boundary (the real
protection is which agent you point at the data - a local/open-weight agent
keeps it in your infra). Masking runs over the string form of each field value;
structure (dict keys, non-string values) is preserved.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Mapping, Sequence

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


#: Regex maskers for format-defined tokens, in apply order: the shapes regex
#: catches cleanly with near-zero false positives (email, secrets, IBAN, IP,
#: MAC, SSN, Luhn-validated card, phone). The broad phone matcher runs last so
#: the more specific digit patterns claim their spans first. Linguistic PII
#: (names, addresses) has no fixed shape and is NOT covered - that is the agent's
#: concern, handled by pointing a local agent at the data, not by STROT masking.
DEFAULT_MASKERS: List[Masker] = [
    mask_email, mask_secret, mask_iban, mask_mac, mask_ip,
    mask_ssn, mask_credit_card, mask_phone,
]


def default_maskers() -> List[Masker]:
    """The regex detection baseline `pii_scan` uses (no model, no network).

    Returns the format-defined-token maskers. STROT embeds records locally and
    sends them nowhere, so there is no boundary to mask at; this is an advisory
    pass that lets an agent see what structured PII a record carries before it
    feeds that record to its own LLM.
    """
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
