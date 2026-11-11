"""Unit tests for PII masking at the teacher boundary (strot.pii).

Pure-Python: the default regex redactors catch the obvious patterns, and the
compiler masks sample records before the teacher ever sees them.
"""

import pytest

from strot import pii
from strot.exceptions import RouterConfigurationError

try:
    import gliner  # noqa: F401
    HAS_GLINER = True
except ImportError:
    HAS_GLINER = False


class TestDefaultMaskers:
    def test_redacts_email(self):
        assert pii.mask_text("reach me at jane.doe@example.com now") == (
            "reach me at [EMAIL] now"
        )

    def test_redacts_ssn(self):
        assert "[SSN]" in pii.mask_text("ssn 123-45-6789 on file")
        assert "123-45-6789" not in pii.mask_text("ssn 123-45-6789 on file")

    def test_redacts_credit_card(self):
        out = pii.mask_text("card 4111 1111 1111 1111 charged")
        assert "[CARD]" in out
        assert "4111" not in out

    def test_redacts_phone(self):
        out = pii.mask_text("call +1 415-555-0132 today")
        assert "[PHONE]" in out
        assert "555" not in out

    def test_clean_text_untouched(self):
        clean = "double billing charge on the pro tier"
        assert pii.mask_text(clean) == clean

    def test_multiple_patterns_in_one_string(self):
        out = pii.mask_text("mail a@b.com or ssn 111-22-3333")
        assert "[EMAIL]" in out and "[SSN]" in out


class TestMaskRecords:
    def test_masks_string_fields_preserves_structure(self):
        records = [
            {"id": 1, "body": "email a@b.com", "amount": 42.0, "ok": True},
        ]
        out = pii.mask_records(records)
        assert out == [{"id": 1, "body": "email [EMAIL]", "amount": 42.0, "ok": True}]
        # original is not mutated
        assert records[0]["body"] == "email a@b.com"

    def test_non_string_values_untouched(self):
        out = pii.mask_records([{"n": 123456789012345, "x": None}])
        assert out == [{"n": 123456789012345, "x": None}]

    def test_custom_maskers_replace_defaults(self):
        redact_all = lambda s: "[X]"
        out = pii.mask_records([{"body": "a@b.com"}], maskers=[redact_all])
        assert out == [{"body": "[X]"}]

    def test_empty_masker_list_is_passthrough(self):
        out = pii.mask_records([{"body": "a@b.com"}], maskers=[])
        assert out == [{"body": "a@b.com"}]


class TestExpandedMaskers:
    def test_default_masker_count(self):
        assert len(pii.DEFAULT_MASKERS) == 8

    def test_redacts_ipv4(self):
        out = pii.mask_text("client 192.168.1.42 connected")
        assert "[IP]" in out and "192.168" not in out

    def test_redacts_ipv6(self):
        out = pii.mask_text("addr 2001:0db8:85a3:0000:0000:8a2e:0370:7334 seen")
        assert "[IP]" in out

    def test_redacts_mac(self):
        out = pii.mask_text("device 00:1A:2B:3C:4D:5E online")
        assert "[MAC]" in out and "1A:2B" not in out

    def test_redacts_iban(self):
        out = pii.mask_text("pay to GB82WEST12345698765432 please")
        assert "[IBAN]" in out and "WEST" not in out

    def test_redacts_secrets(self):
        assert "[SECRET]" in pii.mask_text("key AKIAIOSFODNN7EXAMPLE used")
        assert "[SECRET]" in pii.mask_text("token sk-abcdefghijklmnopqrstuvwxyz0123")
        assert "[SECRET]" in pii.mask_text("auth Bearer abcdef1234567890xyz")

    def test_luhn_validation_is_precise(self):
        # a valid Luhn card IS redacted
        assert "[CARD]" in pii.mask_text("card 4111111111111111 ok")
        # a 16-digit run that FAILS Luhn is left alone (not a real card)
        assert "[CARD]" not in pii.mask_text("order 4111111111111112 ref")


class TestGlinerMasker:
    def test_default_pii_entities_cover_unstructured(self):
        # regex can't do these; GLiNER is the reason they're listed
        assert "person" in pii.DEFAULT_PII_ENTITIES
        assert "address" in pii.DEFAULT_PII_ENTITIES

    @pytest.mark.skipif(HAS_GLINER, reason="gliner present: extra-guard not exercised")
    def test_gliner_masker_requires_extra(self):
        with pytest.raises(RouterConfigurationError):
            pii.gliner_masker()

    @pytest.mark.skipif(HAS_GLINER, reason="gliner present: regex fallback not exercised")
    def test_default_maskers_warns_and_falls_back_to_regex_without_extra(self):
        # GLiNER is the default; without the [pii] extra the default masker list
        # must warn loudly and degrade to the regex pre-pass alone.
        with pytest.warns(UserWarning, match="GLiNER not installed"):
            maskers = pii.default_maskers()
        assert maskers == pii.DEFAULT_MASKERS

    @pytest.mark.skipif(not HAS_GLINER, reason="gliner absent: GLiNER default not testable")
    def test_default_maskers_runs_gliner_after_regex_prepass_when_installed(self):
        # GLiNER is the DEFAULT when the extra is installed: the regex pre-pass
        # runs FIRST (clean format-defined tokens), GLiNER LAST (names/addr/org).
        maskers = pii.default_maskers()
        assert len(maskers) == len(pii.DEFAULT_MASKERS) + 1
        assert maskers[: len(pii.DEFAULT_MASKERS)] == list(pii.DEFAULT_MASKERS)


def test_compiler_masks_records_before_teacher(monkeypatch):
    """End-to-end: the raw PII must never appear in any prompt the compiler
    sends to the teacher LLM; the masked token must."""
    from strot.compiler import StrotCompiler

    registry = {
        "billing": "Billing disputes, charges, and refunds",
        "defect": "Crashes, bugs, and broken behavior",
    }
    sample = [
        {"ticket_id": 1, "body": "double billing charge; email jane@acme.io"},
        {"ticket_id": 2, "body": "app crash defect; ssn 123-45-6789"},
    ]
    paths = (
        "```python\n"
        'ROUTING_LABELS = {"billing": "billing", "defect": "defect"}\n\n\n'
        "def route_context(record: dict) -> str:\n"
        '    return str(record.get("body", ""))\n\n\n'
        "def execute_billing(record: dict) -> dict:\n    return {}\n\n\n"
        "def execute_defect(record: dict) -> dict:\n    return {}\n```"
    )
    teacher = '```python\nLABELS = ["billing", "defect"]\n```'

    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        # First call is synthesis, second is the teacher labeling.
        return teacher if "STROT teacher" in prompt else paths

    compiler = StrotCompiler(nlu_model="SupersonicLabs/Julia-1", llm_client=llm)
    # measure_labels forces the teacher labeling pass without needing torch.
    monkeypatch.setattr(
        StrotCompiler, "_measure_labelings",
        lambda *a, **k: ("synthesized", {}, {"synthesized": 1.0}),
    )
    compiler.compile(sample, "Triage tickets", registry, measure_labels=True)

    teacher_prompts = [p for p in prompts if "STROT teacher" in p]
    assert teacher_prompts, "teacher was never called"
    joined = "\n".join(teacher_prompts)
    assert "jane@acme.io" not in joined
    assert "123-45-6789" not in joined
    assert "[EMAIL]" in joined and "[SSN]" in joined
