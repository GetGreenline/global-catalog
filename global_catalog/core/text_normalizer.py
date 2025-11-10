import re
import unicodedata

_whitespace_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\s]")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _punct_re.sub(" ", s)
    s = _whitespace_re.sub(" ", s)
    return s.strip()

def join_path(*parts: str) -> str:
    parts = [normalize_text(p) for p in parts if p is not None and str(p).strip() != ""]
    return " / ".join(parts)
