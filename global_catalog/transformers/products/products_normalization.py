import re
import html
import json
import unicodedata
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

def _compile_phrase_patterns(phrases: List[str]) -> List[Tuple[str, re.Pattern]]:
    patterns: List[Tuple[str, re.Pattern]] = []
    sep = r"[-\s]+"
    for phrase in phrases:
        tokens = phrase.split()
        if len(tokens) == 1:
            pattern = re.compile(rf"\b{re.escape(tokens[0])}\b")
        else:
            pattern = re.compile(rf"\b{sep.join(map(re.escape, tokens))}\b")
        patterns.append((phrase, pattern))
    return patterns


EXTRACT_TYPE_PATTERNS = _compile_phrase_patterns(
    [
        "live resin",
        "rosin",
        "distillate",
        "diamonds",
        "wax",
        "shatter",
        "badder",
        "sauce",
    ]
)

STRAIN_OR_FLAVOR_PATTERNS = _compile_phrase_patterns(
    [
        "tropic thunder",
        "gush mints",
        "blue dream",
        "sour diesel",
        "wedding cake",
        "girl scout cookies",
        "pineapple express",
        "strawberry cough",
        "purple punch",
        "lemon haze",
        "grape ape",
        "cherry pie",
        "banana og",
        "og kush",
        "gsc",
        "zkittlez",
        "runtz",
        "gushers",
        "tropicana",
        "gelato",
        "mimosa",
        "blueberry",
        "mango",
        "strawberry",
        "lemon",
        "pineapple",
        "mint",
        "vanilla",
    ]
)

PRODUCT_LINE_PATTERNS = _compile_phrase_patterns(
    [
        "select classics",
        "black label",
        "white label",
        "elite",
        "essentials",
        "classics",
        "signature",
        "reserve",
        "limited",
        "platinum",
        "gold",
    ]
)

class ProductNormalizer:
    def __init__(
        self,
        keep_columns: Optional[List[str]] = None,
        name_synonyms: Optional[Dict[str, str]] = None,
        uom_synonyms: Optional[Dict[str, str]] = None,
        brand_aliases: Optional[Dict[str, str]] = None,
        corp_suffixes: Optional[List[str]] = None,
        oz_mg: float = 28000.0,
        fl_oz_mg: float = 29573.5,
        ml_mg: float = 1000.0,
    ):
        self.name_synonyms = name_synonyms or {}
        self.uom_synonyms = uom_synonyms or {}
        self.brand_aliases = {k.lower(): v.lower() for k, v in (brand_aliases or {}).items()}
        self.corp_suffixes = set((corp_suffixes or ["inc","co","corp","corporation","llc","ltd","gmbh","s.a.","sa","bv","ag","plc","pte","oy","srl","spa","k.k.","kk"]))
        self.allowed_uom = {"each", "mg", "gram", "g", "ml", "fl oz", "oz"}
        self._mult = {"mg": 1.0, "gram": 1000.0, "g": 1000.0, "ml": ml_mg, "oz": oz_mg, "fl oz": fl_oz_mg, "each": None}
        self._num_or_fr = r'(\d+(?:\.\d+)?|\.\d+|\d+/\d+)'
        self.keep_columns = keep_columns or [
            "source",
            "product_id",
            "brand_name_norm",
            "product_name_norm",
            "original_product_name",
            "package_size",
            "variant_weight",
            "description_norm",
            "uom_norm",
            "measure_mg",
            "measure_mg_int",
            "strain_type_norm",
            "states_norm",
            "country_norm",
            "extract_type_norm",
            "strain_or_flavor_norm",
            "product_line_norm",
        ]
        self._uom_pattern = r'(?:mg|milligram|milligrams|g|gram|grams|ml|milliliter|milliliters|fl\.?\s*oz|fl\s*oz|fluid\s*ounce|oz|ounce|ounces)'
        self._pack_word_pattern = r'(?:pack|packs|pk|ct|count)'
        self._pack_keyword_regex = re.compile(r'\b(pack|packs|pk|ct|count|each|ea|per)\b', flags=re.IGNORECASE)

    def _normalize_text(self, s: Any) -> str:
        x = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
        x = html.unescape(x)
        x = re.sub(r"<[^>]+>", " ", x)
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
        x = x.lower().strip()
        x = re.sub(r"https?://\S+|www\.\S+", " ", x)
        x = re.sub(r"[^\w\s]+", " ", x)
        x = re.sub(r"\s+", " ", x).strip()
        return x

    def _apply_synonyms(self, text: str, syns: Dict[str, str]) -> str:
        if not text:
            return text
        out = text
        for k, v in syns.items():
            out = re.sub(rf"\b{re.escape(k.lower())}\b", v.lower(), out)
        out = re.sub(r"\s+", " ", out).strip()
        return out
    def _as_float(self, x: Any) -> Optional[float]:
        try:
            if isinstance(x, str) and "/" in x:
                a, b = x.split("/", 1)
                return float(a) / float(b)
            return float(x)
        except Exception:
            return None

    def _normalize_uom_text(self, u: Any) -> Optional[str]:
        if u is None or (isinstance(u, float) and np.isnan(u)):
            return None
        s = str(u).strip().lower()
        s = self._apply_synonyms(s, self.uom_synonyms) if self.uom_synonyms else s
        s = s.replace("ounces", "ounce").replace("grams", "gram").replace("milliliters", "milliliter")
        if s in {"g", "gram"}:
            return "gram"
        if s == "mg":
            return "mg"
        if s in {"ml", "milliliter"}:
            return "ml"
        if s in {"oz", "ounce"}:
            return "oz"
        if s in {"fl oz", "fl. oz", "fl.oz"}:
            return "fl oz"
        if s == "each":
            return "each"
        return s

    def _parse_from_name(self, name: Any) -> Tuple[Optional[float], Optional[str]]:
        measure, uom, _ = self._parse_from_name_with_span(name)
        return (measure, uom)

    def _parse_from_name_with_span(self, name: Any) -> Tuple[Optional[float], Optional[str], Optional[Tuple[int, int]]]:
        if not isinstance(name, str):
            return (None, None, None)
        text = str(name)
        candidates = self._extract_measure_candidates(text)
        if not candidates:
            return (None, None, None)
        candidates.sort(
            key=lambda c: (
                -c["priority"],
                -(c["mg_value"] if c["mg_value"] is not None else -1),
                c["span"][0],
            )
        )
        best = candidates[0]
        return (best["measure"], best["uom"], best["span"])

    def _extract_measure_candidates(self, text: str) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        pattern = re.compile(
            rf'(?P<value>{self._num_or_fr})\s*(?P<unit>{self._uom_pattern})', flags=re.IGNORECASE
        )
        for match in pattern.finditer(text):
            start, end = match.span()
            measure_val = self._as_float(match.group("value"))
            if measure_val is None:
                continue
            unit_norm = self._normalize_uom_text(match.group("unit"))
            if unit_norm not in self.allowed_uom or unit_norm is None:
                continue
            multiplier = self._get_pack_multiplier(text, start, end)
            priority = 0
            if multiplier is not None and multiplier > 0:
                measure_val = measure_val * multiplier
                if multiplier > 1:
                    priority = 3
            if priority == 0 and self._has_pack_keyword_near(text, start, end):
                priority = 1
            mult = self._mult.get(unit_norm)
            mg_value = None
            if mult is not None and measure_val is not None:
                mg_value = measure_val * mult
            candidates.append(
                {
                    "measure": measure_val,
                    "uom": unit_norm,
                    "span": (start, end),
                    "priority": priority,
                    "mg_value": mg_value,
                }
            )
        return candidates

    def _has_pack_keyword_near(self, text: str, start: int, end: int) -> bool:
        window_start = max(0, start - 15)
        window_end = min(len(text), end + 15)
        before = text[window_start:start]
        after = text[end:window_end]
        return bool(self._pack_keyword_regex.search(before) or self._pack_keyword_regex.search(after))

    def _get_pack_multiplier(self, text: str, start: int, end: int) -> Optional[float]:
        prefix = text[max(0, start - 25):start]
        suffix = text[end:min(len(text), end + 25)]
        prefix = prefix.strip(" \t\r\n-_/()[]{}")
        suffix = suffix.strip(" \t\r\n-_/()[]{}")
        prefix = re.sub(r'[\[\](){}]', ' ', prefix).strip()
        suffix = re.sub(r'[\[\](){}]', ' ', suffix).strip()
        count = self._match_prefix_count(prefix)
        if count is None:
            count = self._match_suffix_count(suffix)
        if count is None or count <= 0:
            return None
        return count

    def _match_prefix_count(self, prefix: str) -> Optional[float]:
        if not prefix:
            return None
        patterns = [
            rf'(?P<count>{self._num_or_fr})\s*(?:x|×)\s*$',
            rf'(?P<count>{self._num_or_fr})(?:\s*|-)?{self._pack_word_pattern}\b(?:\s*(?:of|each|ea|per))?\s*$',
            rf'{self._pack_word_pattern}\b\s*(?:of\s*)?(?P<count>{self._num_or_fr})\s*$',
        ]
        for pat in patterns:
            match = re.search(pat, prefix, flags=re.IGNORECASE)
            if match:
                count = self._as_float(match.group("count"))
                if count is not None:
                    return count
        return None

    def _match_suffix_count(self, suffix: str) -> Optional[float]:
        if not suffix:
            return None
        patterns = [
            rf'^(?:x|×)\s*(?P<count>{self._num_or_fr})',
            rf'^(?:each|ea|per)?\s*(?P<count>{self._num_or_fr})(?:\s*|-)?{self._pack_word_pattern}\b',
            rf'^(?:each|ea|per)?\s*{self._pack_word_pattern}\b\s*(?:of\s*)?(?P<count>{self._num_or_fr})',
            rf'^(?P<count>{self._num_or_fr})\s*(?:x|×)',
        ]
        for pat in patterns:
            match = re.search(pat, suffix, flags=re.IGNORECASE)
            if match:
                count = self._as_float(match.group("count"))
                if count is not None:
                    return count
        return None

    def _remove_detected_measure_from_name(self, name: Any) -> Any:
        if name is None or (isinstance(name, float) and np.isnan(name)):
            return name
        text = str(name)
        _, _, span = self._parse_from_name_with_span(text)
        if not span:
            return text
        start, end = span
        stripped = (text[:start] + " " + text[end:]).strip()
        stripped = re.sub(r"\s+", " ", stripped)
        return stripped

    def _to_float_or_none(self, x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            try:
                if pd.isna(x):
                    return None
            except Exception:
                pass
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"na", "none", "null", "nan"}:
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _compute_mg(self, measure, uom) -> Optional[str]:
        if uom is None:
            return None
        u = str(uom).strip().lower()
        if u == "":
            return None
        if u == "each":
            return "each"
        mult = self._mult.get(u)
        if mult is None:
            return None
        m = self._to_float_or_none(measure)
        if m is None:
            return None
        mg_int = int(round(m * mult))
        return f"{mg_int}-milligrams"

    def _parse_mg_int_from_text(self, value: Any) -> Optional[int]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float):
            if np.isnan(value):
                return None
            return int(round(value))
        s = str(value).strip().lower()
        if s in {"", "na", "none", "null", "nan", "each"}:
            return None
        s = s.replace(",", "")
        m = re.match(r"^([0-9]*\.?[0-9]+)\s*-\s*milligrams?$", s)
        if m:
            return int(round(float(m.group(1))))
        m = re.match(r"^([0-9]*\.?[0-9]+)\s*(mg|milligrams?)$", s)
        if m:
            return int(round(float(m.group(1))))
        m = re.match(r"^([0-9]*\.?[0-9]+)\s*(g|grams?)$", s)
        if m:
            return int(round(float(m.group(1)) * 1000))
        if re.match(r"^[0-9]*\.?[0-9]+$", s):
            return int(round(float(s)))
        return None

    def _measure_to_mg_int(self, measure: Any, uom: Any, measure_mg: Any = None) -> Optional[int]:
        u = self._normalize_uom_text(uom)
        if u and u in self._mult and self._mult[u] is not None:
            m = self._to_float_or_none(measure)
            if m is not None:
                return int(round(m * self._mult[u]))
        mg_from_measure = self._parse_mg_int_from_text(measure)
        if mg_from_measure is not None:
            return mg_from_measure
        return self._parse_mg_int_from_text(measure_mg)

    def _first_phrase_match(self, text: str, patterns: List[Tuple[str, re.Pattern]]) -> Optional[str]:
        if not text:
            return None
        for label, pattern in patterns:
            if pattern.search(text):
                return label
        return None

    def _derive_disambiguators(self, name_norm: Any, description_norm: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        name = "" if name_norm is None or (isinstance(name_norm, float) and np.isnan(name_norm)) else str(name_norm)
        desc = "" if description_norm is None or (isinstance(description_norm, float) and np.isnan(description_norm)) else str(description_norm)
        text = f"{name} {desc}".strip()
        if not text:
            return (None, None, None)
        extract_type = self._first_phrase_match(text, EXTRACT_TYPE_PATTERNS)
        strain_or_flavor = self._first_phrase_match(text, STRAIN_OR_FLAVOR_PATTERNS)
        product_line = self._first_phrase_match(text, PRODUCT_LINE_PATTERNS)
        return (extract_type, strain_or_flavor, product_line)

    def _maybe_fill_from_name(self, row: pd.Series) -> pd.Series:
        cur_measure = row.get("measure", np.nan)
        if not (pd.isna(cur_measure) or cur_measure is None):
            return row
        parsed_measure, parsed_uom = self._parse_from_name(row.get("product_name", None))
        if parsed_measure is not None:
            row["measure"] = parsed_measure
            if parsed_uom in self.allowed_uom:
                row["uom"] = parsed_uom
        return row

    def _normalize_brand(self, brand: Any) -> Optional[str]:
        if brand is None or (isinstance(brand, float) and np.isnan(brand)):
            return None
        b = str(brand)
        b = unicodedata.normalize("NFKD", b).encode("ascii","ignore").decode("ascii")
        b = b.casefold().strip()
        b = re.sub(r"[™®©]", " ", b)
        b = re.sub(r"[&+]", " and ", b)
        b = re.sub(r"[^\w\s]", " ", b)
        b = re.sub(r"\s+", " ", b).strip()
        if b in self.brand_aliases:
            b = self.brand_aliases[b]
        parts = [p for p in b.split() if p not in self.corp_suffixes]
        b = " ".join(parts).strip()
        return b or None

    def _normalize_strain_type(self, s: Any) -> Optional[str]:
        x = self._normalize_text(s)
        if not x:
            return None
        m = {
            "indica": "indica",
            "sativa": "sativa",
            "hybrid": "hybrid",
            "cbd": "cbd",
        }
        for k, v in m.items():
            if re.search(rf"\b{k}\b", x):
                return v
        return x

    def _normalize_states(self, s: Any) -> Optional[str]:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return None
        txt = str(s)
        tokens = [t for t in re.split(r"[,\|;/\s]+", txt) if t]
        toks = []
        for t in tokens:
            t2 = unicodedata.normalize("NFKD", t).encode("ascii","ignore").decode("ascii")
            t2 = re.sub(r"[^a-zA-Z]", "", t2).upper()
            if not t2:
                continue
            toks.append(t2)
        if not toks:
            return None
        toks = sorted(list(dict.fromkeys(toks)))
        return ",".join(toks)

    def _normalize_country(self, c: Any) -> Optional[str]:
        x = self._normalize_text(c)
        if not x:
            return None
        m = {
            "united states": "US",
            "usa": "US",
            "us": "US",
            "canada": "CA",
            "ca": "CA",
        }
        if x in m:
            return m[x]
        x2 = re.sub(r"[^a-z]", "", x).upper()
        if len(x2) == 2:
            return x2
        return x2 or None

    def _desc_backfill(self, desc_norm: str, name_norm: str, min_len: int = 10) -> str:
        if not desc_norm or len(desc_norm) < min_len:
            return name_norm or desc_norm
        return desc_norm

    def _ensure_cols(self, df: pd.DataFrame, src: str) -> pd.DataFrame:
        cols = [
            "source",
            "product_name",
            "original_product_name",
            "description",
            "measure",
            "uom",
            "package_size",
            "variant_weight",
            "brand_id",
            "brand_name",
            "brand",
            "variant_weight",
            "strain_type",
            "states",
            "country",
        ]
        if src == "weedmaps":
            cols += [
                "product_variant_weight_converted_mg",
                "product_variant_weight_unit",
                "product_variant_weight_value",
                "brand_name_resolved",
            ]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        if "source" not in df.columns or df["source"].isna().all():
            df["source"] = src
        else:
            df["source"] = df["source"].astype(str).str.strip().str.lower().replace("", src)
        return df

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self.keep_columns if c in df.columns]
        return df[keep].copy()

    def _process_hoodie(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._ensure_cols(df.copy(), "hoodie")
        out["uom"] = out["uom"].map(self._normalize_uom_text)
        out = out.apply(self._maybe_fill_from_name, axis=1)
        out["product_name"] = out["product_name"].apply(self._remove_detected_measure_from_name)
        out["measure_mg"] = [self._compute_mg(m, u) for m, u in zip(out.get("measure", []), out.get("uom", []))]
        out["measure_mg_int"] = [
            self._measure_to_mg_int(m, u, mmg)
            for m, u, mmg in zip(out.get("measure", []), out.get("uom", []), out.get("measure_mg", []))
        ]
        out["product_name_norm"] = out["product_name"].apply(self._normalize_text).apply(lambda x: self._apply_synonyms(x, self.name_synonyms))
        desc_source = "description" if "description" in out.columns else ("description_norm" if "description_norm" in out.columns else None)
        if desc_source is None:
            out["description"] = np.nan
            desc_source = "description"
        desc_norm = out[desc_source].apply(self._normalize_text)

        #testing without limitation as min len was to small
        #out["description_norm"] = [self._desc_backfill(d, n) for d, n in zip(desc_norm, out["product_name_norm"])]
        out["description_norm"] = desc_norm
        brand_name = out.get("brand_name", np.nan).replace(r"^\s*$", np.nan, regex=True)
        if "brand" in out.columns:
            brand_alt = out["brand"].replace(r"^\s*$", np.nan, regex=True)
            brand_name = brand_alt.combine_first(brand_name)
        out["brand_name_norm"] = brand_name.apply(self._normalize_brand)
        out["uom_norm"] = out["uom"].map(self._normalize_uom_text)
        out["strain_type_norm"] = out.get("strain_type", np.nan).apply(self._normalize_strain_type)
        out["states_norm"] = out.get("states", np.nan).apply(self._normalize_states)
        out["country_norm"] = out.get("country", np.nan).apply(self._normalize_country)
        disambiguators = [
            self._derive_disambiguators(n, d) for n, d in zip(out.get("product_name_norm", []), out.get("description_norm", []))
        ]
        out["extract_type_norm"] = [d[0] for d in disambiguators]
        out["strain_or_flavor_norm"] = [d[1] for d in disambiguators]
        out["product_line_norm"] = [d[2] for d in disambiguators]
        out["source"] = "hoodie"
        return self._finalize(out)

    def _process_weedmaps(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._ensure_cols(df.copy(), "weedmaps")
        if "brand_name" not in out.columns:
            out["brand_name"] = np.nan
        out["brand_name"] = out["brand_name"].replace(r"^\s*$", np.nan, regex=True)
        if "brand_name_resolved" in out.columns:
            resolved = out["brand_name_resolved"].replace(r"^\s*$", np.nan, regex=True)
            out["brand_name"] = resolved.combine_first(out["brand_name"])
        if "brand" in out.columns:
            brand_alt = out["brand"].replace(r"^\s*$", np.nan, regex=True)
            out["brand_name"] = brand_alt.combine_first(out["brand_name"])

        # Seed measure_mg from product_variant_weight_converted_mg when present (e.g., Redshift CSVs)
        measure_mg = pd.Series([np.nan] * len(out), index=out.index)
        if "product_variant_weight_converted_mg" in out.columns:
            pvw = out["product_variant_weight_converted_mg"].replace(r"^\s*$", np.nan, regex=True)
            measure_mg = pvw.astype(object)
        if "measure" in out.columns and "uom" in out.columns:
            computed = [self._compute_mg(m, u) for m, u in zip(out.get("measure", []), out.get("uom", []))]
            measure_mg = measure_mg.combine_first(pd.Series(computed, index=out.index))
        if "variant_weight" in out.columns:
            vw = out["variant_weight"].replace(r"^\s*$", np.nan, regex=True)
            measure_mg = measure_mg.combine_first(vw)
        out["measure_mg"] = measure_mg.astype(object)
        out["measure_mg_int"] = [
            self._measure_to_mg_int(m, u, mmg)
            for m, u, mmg in zip(out.get("measure", []), out.get("uom", []), out.get("measure_mg", []))
        ]
        u_source = None
        if "uom" in out.columns and out["uom"].notna().any():
            u_source = out["uom"]
        elif "product_variant_weight_unit" in out.columns:
            u_source = out["product_variant_weight_unit"]
        out["uom_norm"] = u_source.map(self._normalize_uom_text) if u_source is not None else None
        out["product_name"] = out["product_name"].apply(self._remove_detected_measure_from_name)
        out["product_name_norm"] = out["product_name"].apply(self._normalize_text).apply(lambda x: self._apply_synonyms(x, self.name_synonyms))
        desc_source = "description" if "description" in out.columns else ("description_norm" if "description_norm" in out.columns else None)
        if desc_source is None:
            out["description"] = np.nan
            desc_source = "description"
        desc_norm = out[desc_source].apply(self._normalize_text)
        out["description_norm"] = desc_norm
        out["brand_name_norm"] = out.get("brand_name", np.nan).apply(self._normalize_brand)
        out["strain_type_norm"] = out.get("strain_type", np.nan).apply(self._normalize_strain_type)
        out["states_norm"] = out.get("states", np.nan).apply(self._normalize_states)
        out["country_norm"] = out.get("country", np.nan).apply(self._normalize_country)
        disambiguators = [
            self._derive_disambiguators(n, d) for n, d in zip(out.get("product_name_norm", []), out.get("description_norm", []))
        ]
        out["extract_type_norm"] = [d[0] for d in disambiguators]
        out["strain_or_flavor_norm"] = [d[1] for d in disambiguators]
        out["product_line_norm"] = [d[2] for d in disambiguators]
        out["source"] = "weedmaps"
        return self._finalize(out)

    def process(
        self,
        hoodie_df: Optional[pd.DataFrame] = None,
        weedmaps_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        if hoodie_df is not None and len(hoodie_df) > 0:
            frames.append(self._process_hoodie(hoodie_df))
        if weedmaps_df is not None and len(weedmaps_df) > 0:
            frames.append(self._process_weedmaps(weedmaps_df))
        if not frames:
            return pd.DataFrame(columns=self.keep_columns)
        out = pd.concat(frames, ignore_index=True, sort=False)
        return out


class EnrichedProductNormalizer(ProductNormalizer):
    """Normalizer for records containing matching_enrichment/original_data objects."""

    def _coerce_dict(self, value: Any) -> Dict[str, Any]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return {}
        return {}

    def _first_non_empty(self, *values: Any) -> Any:
        for v in values:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            if isinstance(v, str):
                if v.strip() == "" or v.strip().lower() in {"null", "none", "nan"}:
                    continue
                return v
            return v
        return None

    def _flatten_enriched(self, df: pd.DataFrame, source_hint: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            me = self._coerce_dict(row.get("matching_enrichment"))
            orig = self._coerce_dict(row.get("original_data"))
            source = self._first_non_empty(me.get("source"), orig.get("source"), source_hint)
            product_id = self._first_non_empty(me.get("product_id"), row.get("product_id"), orig.get("product_id"))
            product_name = self._first_non_empty(
                me.get("product_name"),
                orig.get("product_name"),
                row.get("raw_item_name"),
                row.get("product_name"),
            )
            original_product_name = self._first_non_empty(
                orig.get("product_name"),
                row.get("raw_item_name"),
                me.get("product_name"),
            )
            brand_name = self._first_non_empty(me.get("brand"), orig.get("brand_name"), row.get("brand_name"))
            description = self._first_non_empty(orig.get("description"), row.get("description"))
            measure = self._first_non_empty(me.get("measure"))
            uom = self._first_non_empty(me.get("uom"))
            variant_weight = self._first_non_empty(me.get("variant_weight"))
            package_size = self._first_non_empty(me.get("package_size"))
            strain_type = self._first_non_empty(orig.get("strain_type"), row.get("strain_type"))
            rows.append(
                {
                    "source": source,
                    "product_id": product_id,
                    "product_name": product_name,
                    "original_product_name": original_product_name,
                    "brand_name": brand_name,
                    "description": description,
                    "measure": measure,
                    "uom": uom,
                    "variant_weight": variant_weight,
                    "package_size": package_size,
                    "strain_type": strain_type,
                    "product_variant_weight_converted_mg": orig.get("product_variant_weight_converted_mg"),
                    "product_variant_weight_unit": orig.get("product_variant_weight_unit"),
                    "product_variant_weight_value": orig.get("product_variant_weight_value"),
                }
            )
        return pd.DataFrame(rows)

    def _process_enriched_source(self, df: pd.DataFrame, source_hint: str) -> pd.DataFrame:
        flat = self._flatten_enriched(df, source_hint)
        if source_hint == "weedmaps":
            return self._process_weedmaps(flat)
        return self._process_hoodie(flat)

    def process(
        self,
        hoodie_df: Optional[pd.DataFrame] = None,
        weedmaps_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        if hoodie_df is not None and len(hoodie_df) > 0:
            frames.append(self._process_enriched_source(hoodie_df, "hoodie"))
        if weedmaps_df is not None and len(weedmaps_df) > 0:
            frames.append(self._process_enriched_source(weedmaps_df, "weedmaps"))
        if not frames:
            return pd.DataFrame(columns=self.keep_columns)
        return pd.concat(frames, ignore_index=True, sort=False)
