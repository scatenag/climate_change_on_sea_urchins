"""download._mask: a message that may carry the credentials must not show them, whatever their shape (found by the
independent review of MA.2): one secret inside the other, quoted or URL-encoded forms, very short secrets."""
import pytest

from climate_change_on_sea_urchins.download import Credentials, _mask

WITHHELD = "withheld"


def test_a_password_that_contains_the_username_is_fully_masked():
    out = _mask("user abcdef123 failed", Credentials("abc", "abcdef123"))
    assert "def123" not in out and "abcdef123" not in out


def test_the_longest_secret_goes_first_whichever_order_they_come_in():
    out = _mask("mario1985! and mario", Credentials("mario", "mario1985!"))
    assert "1985" not in out and "mario" not in out


@pytest.mark.parametrize("secret", ["p\\q'r", 'he said "hi"', "pass!w0rd", "user@mail.com", "a b+c", "100%/sure"])
def test_quoted_and_url_encoded_forms_are_masked_too(secret):
    from urllib.parse import quote, quote_plus
    import json
    c = Credentials("some-user-name", secret)
    for shown in (secret, repr(secret)[1:-1], quote(secret, safe=""), quote_plus(secret), json.dumps(secret)[1:-1]):
        out = _mask(f"failed for {shown} at the server", c)
        assert shown not in out or shown in ("", "***"), (shown, out)


def test_a_secret_too_short_to_mask_withholds_the_whole_message():
    out = _mask("failed at 2026 for user 2", Credentials("2", "a-long-enough-password"))
    assert WITHHELD in out and "2026" not in out


def test_a_normal_message_without_the_secrets_is_untouched():
    assert _mask("the product has no data", Credentials("someuser1", "s3cr3t-pa55")) == "the product has no data"


def test_the_secrets_never_appear_in_the_output_of_a_masked_message_with_both_inside():
    c = Credentials("mario.rossi-user", "s3cr3t-pa55-w0rd")
    out = _mask(f"401 for {c.username}:{c.password}@host", c)
    assert c.username not in out and c.password not in out and "***" in out
