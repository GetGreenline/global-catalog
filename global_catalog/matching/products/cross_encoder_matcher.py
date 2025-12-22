"""
Cross-encoder matcher for product pairs.

Scores candidate pairs using a local sentence-transformers CrossEncoder model
trained on text_w/text_h inputs from the data generation pipeline.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _empty_matches() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "left_index",
            "right_index",
            "left_source",
            "right_source",
            "left_product_name",
            "right_product_name",
            "left_brand_name",
            "right_brand_name",
            "similarity",
            "name_score",
            "match_type",
        ]
    )


@dataclass
class CrossEncoderMatcherConfig:
    model_path: str = "artifacts/products/datasets/artifacts/products/models/cross_encoder_v1"
    batch_size: int = 32
    pair_chunk_size: int = 10000
    device: str = "cpu"
    max_length: int = 256
    threshold: Optional[float] = 0.0
    desc_max_chars: int = 800
    text_separator: str = " || "
    use_raw_data: bool = True
    match_type: str = "cross_encoder_v1"
    show_progress_bar: bool = True


class CrossEncoderMatcher:
    """Cross-encoder semantic matcher producing a score per candidate pair."""

    def __init__(self, cfg: Optional[CrossEncoderMatcherConfig] = None):
        self.cfg = cfg or CrossEncoderMatcherConfig()
        self._model = None

    def __call__(
        self,
        df_norm: pd.DataFrame,
        pairs_df: Optional[pd.DataFrame],
        cfg=None,
        *,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        if pairs_df is None or pairs_df.empty:
            return _empty_matches()

        cfg = cfg or self.cfg
        logger.info(
            "CrossEncoderMatcher: scoring %d candidate pairs with model at %s",
            len(pairs_df),
            cfg.model_path,
        )
        left_idx_arr = pairs_df["left_index"].to_numpy()
        right_idx_arr = pairs_df["right_index"].to_numpy()

        unique_idxs = pd.Index(pd.unique(np.concatenate([left_idx_arr, right_idx_arr])))
        df_sub = df_norm.loc[unique_idxs]
        idx_to_pos = {idx: pos for pos, idx in enumerate(df_sub.index)}

        text_series = self._build_text_series(df_sub, raw_data if cfg.use_raw_data else None, cfg)
        texts = text_series.to_numpy()

        name_col = "product_name_norm" if "product_name_norm" in df_sub.columns else "normalized_product_name"
        brand_col = "brand_name_norm" if "brand_name_norm" in df_sub.columns else "brand_name"
        names = df_sub[name_col].fillna("").astype(str).to_numpy()
        brands = df_sub.get(brand_col, pd.Series([""] * len(df_sub), index=df_sub.index)).fillna("").astype(str).to_numpy()
        sources = df_sub.get("source", pd.Series([""] * len(df_sub), index=df_sub.index)).fillna("").astype(str).to_numpy()

        left_pos: List[int] = []
        right_pos: List[int] = []
        left_idx_out: List[Any] = []
        right_idx_out: List[Any] = []
        for left_idx, right_idx in zip(left_idx_arr, right_idx_arr):
            li = idx_to_pos.get(left_idx)
            rj = idx_to_pos.get(right_idx)
            if li is None or rj is None:
                continue
            left_pos.append(li)
            right_pos.append(rj)
            left_idx_out.append(left_idx)
            right_idx_out.append(right_idx)

        if not left_pos:
            return _empty_matches()

        records = []
        threshold = cfg.threshold
        match_type = cfg.match_type
        total = len(left_pos)
        chunk_size = cfg.pair_chunk_size or total
        if chunk_size <= 0:
            chunk_size = total
        if chunk_size < total:
            logger.info("CrossEncoderMatcher: scoring in chunks of %d pairs", chunk_size)

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            scores = self._score_pairs(
                texts,
                left_pos[start:end],
                right_pos[start:end],
                cfg,
            )
            for lp, rp, li, rj, score in zip(
                left_pos[start:end],
                right_pos[start:end],
                left_idx_out[start:end],
                right_idx_out[start:end],
                scores,
            ):
                if threshold is not None and score < threshold:
                    continue
                records.append(
                    {
                        "left_index": int(li),
                        "right_index": int(rj),
                        "left_source": sources[lp],
                        "right_source": sources[rp],
                        "left_product_name": names[lp],
                        "right_product_name": names[rp],
                        "left_brand_name": brands[lp],
                        "right_brand_name": brands[rp],
                        "similarity": float(score),
                        "name_score": float(score),
                        "match_type": match_type,
                    }
                )

        if not records:
            return _empty_matches()

        return pd.DataFrame(records).sort_values("similarity", ascending=False).reset_index(drop=True)

    # --- Text prep ------------------------------------------------------

    def _build_text_series(
        self,
        df_sub: pd.DataFrame,
        raw_data: Optional[Dict[str, Any]],
        cfg: CrossEncoderMatcherConfig,
    ) -> pd.Series:
        raw_lookup = self._build_raw_lookup(raw_data) if raw_data else {}
        texts = []
        for _, row in df_sub.iterrows():
            raw_row = self._get_raw(raw_lookup, row.get("source"), row.get("product_id")) if raw_lookup else {}
            texts.append(self._row_to_text(row, raw_row, cfg))
        return pd.Series(texts, index=df_sub.index)

    def _row_to_text(self, row: pd.Series, raw_row: Dict[str, Any], cfg: CrossEncoderMatcherConfig) -> str:
        brand_raw = raw_row.get("brand_name") if raw_row else None
        product_raw = raw_row.get("product_name") if raw_row else None
        desc_raw = raw_row.get("description") if raw_row else None

        brand = self._first_non_empty(brand_raw, row.get("brand_name"), row.get("brand_name_norm"))
        name = self._first_non_empty(product_raw, row.get("product_name"), row.get("product_name_norm"))
        desc = self._first_non_empty(desc_raw, row.get("description"), row.get("description_norm"))
        uom = self._first_non_empty(row.get("uom_norm"), row.get("uom"))
        measure = self._first_non_empty(row.get("measure_mg"), row.get("measure"))

        if isinstance(desc, str) and cfg.desc_max_chars and len(desc) > cfg.desc_max_chars:
            desc = desc[: cfg.desc_max_chars]

        parts = [self._norm_text(p) for p in (brand, name, measure, uom) if self._norm_text(p)]
        text = " ".join(parts)
        if desc:
            desc_txt = self._norm_text(desc)
            if text:
                text = f"{text}{cfg.text_separator}{desc_txt}"
            else:
                text = desc_txt
        return text

    @staticmethod
    def _norm_text(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null"}:
            return ""
        return text

    def _first_non_empty(self, *values: Any) -> str:
        for value in values:
            text = self._norm_text(value)
            if text:
                return text
        return ""

    # --- Model and scoring ---------------------------------------------

    def _score_pairs(
        self,
        texts: np.ndarray,
        left_pos: List[int],
        right_pos: List[int],
        cfg: CrossEncoderMatcherConfig,
    ) -> np.ndarray:
        model = self._load_model(cfg)
        scores: List[float] = []
        total = len(left_pos)
        batch_size = max(1, int(cfg.batch_size))
        t0 = time.perf_counter()
        log_every = max(1, total // 10)
        next_log = log_every
        processed = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            pairs = list(zip(texts[left_pos[start:end]], texts[right_pos[start:end]]))
            show_bar = cfg.show_progress_bar and total <= batch_size
            batch_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=show_bar)
            scores.extend([float(s) for s in batch_scores])
            processed += (end - start)
            if processed >= next_log and total > batch_size:
                logger.info("CrossEncoderMatcher: scored %d/%d pairs", processed, total)
                next_log += log_every

        logger.info(
            "CrossEncoderMatcher: scored %d pairs in %.2fs",
            total,
            time.perf_counter() - t0,
        )
        return np.array(scores, dtype=float)

    def _load_model(self, cfg: CrossEncoderMatcherConfig):
        if self._model is not None:
            return self._model

        model_dir = self._resolve_model_dir(Path(cfg.model_path))
        from sentence_transformers.cross_encoder import CrossEncoder

        logger.info("CrossEncoderMatcher: loading model from %s", model_dir)
        self._model = CrossEncoder(
            str(model_dir),
            num_labels=1,
            max_length=cfg.max_length,
            device=cfg.device,
        )
        return self._model

    @staticmethod
    def _resolve_model_dir(model_dir: Path) -> Path:
        if model_dir.is_file():
            return model_dir

        has_config = (model_dir / "config.json").exists()
        has_weights = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
        if has_config and has_weights:
            return model_dir

        compat_dir = model_dir / "_compat"
        compat_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CrossEncoderMatcher: preparing compat model dir at %s", compat_dir)

        mapping = {
            "config.json": ["config.json", "config1.json"],
            "tokenizer.json": ["tokenizer.json", "tokenizer1.json"],
            "tokenizer_config.json": ["tokenizer_config.json", "tokenizer_config1.json"],
            "special_tokens_map.json": ["special_tokens_map.json", "special_tokens_map1.json"],
            "model.safetensors": ["model.safetensors", "model1.safetensors"],
            "pytorch_model.bin": ["pytorch_model.bin", "model.bin"],
            "vocab.txt": ["vocab.txt"],
        }

        for dest, candidates in mapping.items():
            dest_path = compat_dir / dest
            if dest_path.exists():
                continue
            for candidate in candidates:
                src_path = model_dir / candidate
                if src_path.exists():
                    shutil.copyfile(src_path, dest_path)
                    break

        return compat_dir

    # --- Raw lookup helpers --------------------------------------------

    @staticmethod
    def _build_raw_lookup(raw_data: Optional[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        lookup: Dict[str, pd.DataFrame] = {}
        if not isinstance(raw_data, dict):
            return lookup
        for src, df in raw_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if "product_id" not in df.columns:
                continue
            src_key = str(src).lower()
            tmp = df.copy()
            tmp["__product_id_key__"] = tmp["product_id"].astype(str)
            lookup[src_key] = tmp.set_index("__product_id_key__", drop=False)
        return lookup

    @staticmethod
    def _get_raw(lookup: Dict[str, pd.DataFrame], source: Any, product_id: Any) -> Dict[str, Any]:
        if source is None or product_id is None:
            return {}
        table = lookup.get(str(source).lower())
        if table is None or table.empty:
            return {}
        key = str(product_id)
        try:
            row = table.loc[key]
        except Exception:
            return {}
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        data = row.to_dict()
        data.pop("__product_id_key__", None)
        return data
