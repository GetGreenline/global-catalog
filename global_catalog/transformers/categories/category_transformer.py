from abc import ABC, abstractmethod
import pandas as pd
import hashlib
import unicodedata
import re


CONTRACT_COLUMNS = [
    "external_id", "category_id", "source",
    "level_one", "level_two", "level_three",
    "l1_norm", "l2_norm", "l3_norm",
    "path_norm", "path_id",
    "id",
    "updated_at",
]


def _ascii_lower(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    return re.sub(r"\s+"," ", s.strip().lower())

def _md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _build_id(source: str, l1: str, l2: str, l3: str) -> str:
    return _md5_hex(f"v1|{_ascii_lower(source)}|{_ascii_lower(l1)}|{_ascii_lower(l2)}|{_ascii_lower(l3)}")


def _build_path_id(l1: str, l2: str, l3: str) -> str:
    return _md5_hex(f"{l1}/{l2}/{l3}")


def _build_id(source: str, l1_raw: str, l2_raw: str, l3_raw: str) -> str:
    l1b = _ascii_lower(l1_raw)
    l2b = _ascii_lower(l2_raw)
    l3b = _ascii_lower(l3_raw)
    return _md5_hex(f"v1|{_ascii_lower(source)}|{l1b}|{l2b}|{l3b}")


class BaseCategoryTransformer(ABC):


    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def _ensure_contract_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        defaults = {
            "external_id": "", "category_id": "", "source": "",
            "level_one": "", "level_two": "", "level_three": "",
            "l1_norm": "", "l2_norm": "", "l3_norm": "",
            "path_norm": "", "path_id": "",
            "id": "",
        }
        for c, v in defaults.items():
            if c not in out.columns:
                out[c] = v
        if "updated_at" not in out.columns:
            out["updated_at"] = pd.NaT
        for c in CONTRACT_COLUMNS:
            if c != "updated_at":
                out[c] = out[c].astype(str).fillna("")
        out["updated_at"] = pd.to_datetime(out["updated_at"], errors="coerce")
        return out[CONTRACT_COLUMNS]

    def _standardize(
        self,
        df: pd.DataFrame,
        *,
        source: str,
        external_id_col: str | None,
        category_id_col: str | None,
    ) -> pd.DataFrame:
        out = df.copy()
        if external_id_col and external_id_col in out.columns:
            out["external_id"] = out[external_id_col].astype(str).fillna("").str.strip()
        else:
            out["external_id"] = ""
        if category_id_col and category_id_col in out.columns:
            out["category_id"] = out[category_id_col].astype(str).fillna("").str.strip()
        else:
            out["category_id"] = ""
        out["source"] = str(source).strip().lower()
        for c in ("level_one", "level_two", "level_three"):
            if c not in out.columns:
                out[c] = ""
            out[c] = out[c].fillna("")
        # out["l1_norm"] = out["level_one"].map(_norm_field)
        # out["l2_norm"] = out["level_two"].map(_norm_field)
        # out["l3_norm"] = out["level_three"].map(_norm_field)
        out["path_norm"] = out["l1_norm"] + "/" + out["l2_norm"] + "/" + out["l3_norm"]
        out["path_id"] = out["path_norm"].map(_md5_hex)
        out["id"] = out.apply(
            lambda r: _build_id(r["source"], r["level_one"], r["level_two"], r["level_three"]),
            axis=1
        )
        out["updated_at"] = pd.to_datetime(out.get("updated_at"), errors="coerce")
        return self._ensure_contract_cols(out)


class HoodieCategoryTransformer(BaseCategoryTransformer):


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["category_id", "level_one", "level_two", "level_three", "updated_at"]
        subset = df[cols].copy()
        return self._standardize(
            subset,
            source="hoodie",
            external_id_col=None,
            category_id_col="category_id",
        )


class WeedmapsCategoryTransformer(BaseCategoryTransformer):

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["external_id", "level_one", "level_two", "level_three", "updated_at"]
        subset = df[[c for c in cols if c in df.columns]].copy()
        return self._standardize(
            subset,
            source="weedmaps",
            external_id_col="external_id",
            category_id_col=None,
        )
