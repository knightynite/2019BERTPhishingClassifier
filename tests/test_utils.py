"""Tests for src.utils — text neutralization + URL feature extraction."""
import unittest

from src.utils import (
    URL_RE,
    IP_RE,
    EMAIL_ADDR_RE,
    decode_subject,
    neutralize,
    url_features,
    normalize_email,
)


class TestRegexes(unittest.TestCase):
    def test_url_regex_matches_http_and_https(self):
        self.assertTrue(URL_RE.search("see http://x.com/a"))
        self.assertTrue(URL_RE.search("see HTTPS://X.COM/path?q=1"))

    def test_url_regex_does_not_match_bare_words(self):
        self.assertFalse(URL_RE.search("just words here, no url"))

    def test_ip_regex(self):
        self.assertTrue(IP_RE.search("from 192.168.1.1 today"))
        self.assertFalse(IP_RE.search("v1.2.3 release"))

    def test_email_regex(self):
        self.assertTrue(EMAIL_ADDR_RE.search("contact ops+abuse@example.com"))


class TestNeutralize(unittest.TestCase):
    def test_replaces_urls(self):
        out = neutralize("click http://bad.example/path or http://other.example")
        self.assertNotIn("bad.example", out)
        self.assertIn("<URL>", out)

    def test_replaces_ips(self):
        out = neutralize("from 10.0.0.1 came a message")
        self.assertIn("<IP>", out)

    def test_collapses_whitespace(self):
        out = neutralize("hello    world\n\nhi")
        self.assertEqual(out, "hello world hi")


class TestURLFeatures(unittest.TestCase):
    def test_no_urls(self):
        f = url_features("plain text")
        self.assertEqual(f['n_urls'], 0)
        self.assertFalse(f['has_ip_url'])

    def test_ip_url_detected(self):
        f = url_features("http://192.168.0.1/login")
        self.assertEqual(f['n_urls'], 1)
        self.assertTrue(f['has_ip_url'])

    def test_mixed_protocol(self):
        f = url_features("http://a.com and https://b.com")
        self.assertTrue(f['mixed_protocol'])


class TestDecodeSubject(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(decode_subject("Hello"), "Hello")

    def test_quoted_printable(self):
        # =?utf-8?Q?H=C3=A9llo?= → "Héllo"
        self.assertEqual(decode_subject("=?utf-8?Q?H=C3=A9llo?="), "Héllo")


class TestNormalizeEmail(unittest.TestCase):
    def test_strips_headers_and_neutralizes(self):
        raw = (
            "From: bad@evil.example\nSubject: Click\n\n"
            "Hello, click http://phish.example/login now from 1.2.3.4."
        )
        out = normalize_email(raw)
        self.assertIn("<URL>", out)
        self.assertIn("<IP>", out)
        self.assertNotIn("From:", out)


if __name__ == '__main__':
    unittest.main()
