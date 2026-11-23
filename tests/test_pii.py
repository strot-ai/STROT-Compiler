"""Unit tests for PII masking at the teacher boundary (strot.pii).

Pure-Python: the default regex redactors catch the obvious patterns, and the
compiler masks sample records before the teacher ever sees them.
"""

from strot import pii


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


def test_default_maskers_is_the_regex_baseline(self=None):
    # GLiNER is gone: default_maskers() is the regex detection list, no model.
    assert pii.default_maskers() == list(pii.DEFAULT_MASKERS)
