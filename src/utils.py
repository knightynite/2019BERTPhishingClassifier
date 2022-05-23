"""Text preprocessing utilities for the phishing classifier.

Email-specific normalization: strip headers, decode quoted-printable,
neutralize URLs (replace with `<URL>`) and IPs (replace with `<IP>`).
The goal is to keep the structural signal (e.g. "click here", URL pattern,
mismatched display vs. href) while not letting the model overfit to
specific URL strings in the training set.
"""
import re
from email import message_from_string
from email.header import decode_header


URL_RE = re.compile(r'https?://[^\s<>"\'`]+', re.IGNORECASE)
IP_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
EMAIL_ADDR_RE = re.compile(r'[\w\.\+-]+@[\w\.-]+\.[a-zA-Z]{2,}')
WS_RE = re.compile(r'\s+')


def strip_headers(raw: str) -> str:
    """Return only the message body from an RFC 822 string."""
    try:
        msg = message_from_string(raw)
    except Exception:
        return raw
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype.startswith('text/'):
                payload = part.get_payload(decode=True)
                if payload:
                    parts.append(_safe_decode(payload))
        return '\n'.join(parts)
    payload = msg.get_payload(decode=True)
    if payload is None:
        return msg.get_payload() or ''
    return _safe_decode(payload)


def _safe_decode(raw: bytes) -> str:
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


def decode_subject(s: str) -> str:
    """Decode RFC 2047 encoded-word headers (=?utf-8?Q?...?=)."""
    parts = decode_header(s)
    out = []
    for txt, enc in parts:
        if isinstance(txt, bytes):
            out.append(txt.decode(enc or 'utf-8', errors='replace'))
        else:
            out.append(txt)
    return ''.join(out)


def neutralize(text: str, replace_urls=True, replace_ips=True,
               replace_emails=True) -> str:
    """Replace URLs/IPs/email addresses with placeholder tokens."""
    if replace_urls:
        text = URL_RE.sub(' <URL> ', text)
    if replace_ips:
        text = IP_RE.sub(' <IP> ', text)
    if replace_emails:
        text = EMAIL_ADDR_RE.sub(' <EMAIL> ', text)
    return WS_RE.sub(' ', text).strip()


def url_features(text: str) -> dict:
    """Surface-level URL-based features that a classifier could attend to."""
    urls = URL_RE.findall(text)
    if not urls:
        return {'n_urls': 0, 'has_ip_url': False, 'max_url_len': 0,
                'mixed_protocol': False}
    has_ip = any(IP_RE.search(u) for u in urls)
    protocols = set(u.split(':', 1)[0].lower() for u in urls)
    return {
        'n_urls': len(urls),
        'has_ip_url': has_ip,
        'max_url_len': max(len(u) for u in urls),
        'mixed_protocol': len(protocols) > 1,
    }


def normalize_email(raw: str) -> str:
    """Full pipeline: headers → body → neutralize → collapse whitespace."""
    body = strip_headers(raw)
    return neutralize(body)
