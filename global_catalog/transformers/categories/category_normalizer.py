# global_catalog/transformers/categories/category_normalizer.py

from __future__ import annotations

import os
import re
import yaml
import hashlib
import pandas as pd
from typing import Any

from global_catalog.core.text_normalizer import normalize_text, join_path


def _ascii_lower(val: Any) -> str:

    if val is None:
        s = ""
    else:
        s = str(val)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _build_row_id(source: str, l1_raw: str, l2_raw: str, l3_raw: str) -> str:

    return _md5_hex(
        f"v1|{_ascii_lower(source)}|{_ascii_lower(l1_raw)}|{_ascii_lower(l2_raw)}|{_ascii_lower(l3_raw)}"
    )


class CategoryNormalizer:


    def __init__(self, synonyms_path: str | None = None):

        self.l1_synonyms = {"pre roll": "pre-roll", "vapes": "vape pen"}
        if synonyms_path and os.path.exists(synonyms_path):
            with open(synonyms_path, "r") as f:
                data = yaml.safe_load(f) or {}
            self.l1_synonyms.update(data.get("l1", {}))

    def _apply_synonyms(self, text: str, synonym_map: dict) -> str:
        if not isinstance(text, str):
            return text
        return synonym_map.get(text.lower(), text)

    def normalize_plurals(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        word = text.strip()
        blocklist = {"glass", "cannabis", "press", "mass", "boss", "class"}
        if word.lower() in blocklist:
            return word
        if re.match(r".{4,}s$", word) and not re.search(r"(ss|is|us)$", word, flags=re.I):
            return word[:-1]
        return word

    def handle_unspecified(self, row: pd.Series) -> pd.Series:
        l2 = row.get("l2_norm")
        if isinstance(l2, str):
            row["l2_norm"] = re.sub(r"\(\s*unspecified\s*\)", "", l2, flags=re.I).strip()
        return row

    def handle_infused(self, row: pd.Series) -> pd.Series:
        l1 = row.get("l1_norm")
        if isinstance(l1, str) and l1.lower().startswith("infused "):
            row["l1_norm"] = l1[8:].strip()
            row["is_infused"] = True
        return row

    def process(self, df: pd.DataFrame) -> pd.DataFrame:

        out = df.copy()

        for c in ("source", "level_one", "level_two", "level_three"):
            if c not in out.columns:
                out[c] = pd.NA

        out[["level_one", "level_two", "level_three"]] = (
            out[["level_one", "level_two", "level_three"]].fillna("uncategorized")
        )

        out["l1_norm"] = (
            out["level_one"]
            .apply(normalize_text)
            .apply(lambda x: self._apply_synonyms(x, self.l1_synonyms))
            .apply(self.normalize_plurals)
        )
        out["l2_norm"] = out["level_two"].apply(normalize_text).apply(self.normalize_plurals)
        out["l3_norm"] = out["level_three"].apply(normalize_text).apply(self.normalize_plurals)

        out["is_infused"] = False
        out = out.apply(self.handle_unspecified, axis=1)
        out = out.apply(self.handle_infused, axis=1)

        out["path_norm"] = out.apply(
            lambda r: join_path(r["l1_norm"], r["l2_norm"], r["l3_norm"]), axis=1
        )

        if "id" not in out.columns:
            out["id"] = ""
        need_id = out["id"].isna() | (out["id"].astype(str).str.strip() == "")
        if need_id.any():
            out.loc[need_id, "id"] = out.loc[need_id].apply(
                lambda r: _build_row_id(
                    r.get("source_raw", r.get("source", "")),
                    r.get("level_one", ""),
                    r.get("level_two", ""),
                    r.get("level_three", ""),
                ),
                axis=1,
            )

        return out
