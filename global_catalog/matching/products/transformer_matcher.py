"""
Transformer-based matcher for product pairs using sentence-transformers.

This module mirrors the fuzzy matcher contract so it can be swapped into
`ProductPipeline.match` without changing downstream resolution logic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


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
class TransformerMatcherConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64
    device: str = "cpu"
    name_weight: float = 1.0
    desc_weight: float = 1.0
    max_desc_tokens: int = 128
    normalize_embeddings: bool = True
    threshold: float = 0.7
    use_fp16: bool = True
    cache_dir: str = "artifacts/products/embeddings"


class TransformerMatcher:
    """Bi-encoder semantic matcher producing cosine similarity per candidate pair."""

    def __init__(self, cfg: Optional[TransformerMatcherConfig] = None):
        self.cfg = cfg or TransformerMatcherConfig()
        self._model = None

    # --- Public API -----------------------------------------------------

    def __call__(self, df_norm: pd.DataFrame, pairs_df: Optional[pd.DataFrame], cfg=None) -> pd.DataFrame:
        if pairs_df is None or pairs_df.empty:
            return _empty_matches()

        cfg = cfg or self.cfg
        name_col = "product_name_norm" if "product_name_norm" in df_norm.columns else "normalized_product_name"
        desc_col = "description_norm" if "description_norm" in df_norm.columns else "description"
        brand_col = "brand_name_norm" if "brand_name_norm" in df_norm.columns else "brand_name"

        # Build text inputs and embeddings (with optional cache).
        text_series = self._build_text_inputs(df_norm, name_col, desc_col, cfg)
        embeddings, idx_to_pos = self._get_embeddings(text_series, cfg)

        # Arrays aligned to df_norm index order for quick lookup.
        names = df_norm[name_col].fillna("").astype(str).to_numpy()
        brands = df_norm.get(brand_col, pd.Series([""] * len(df_norm), index=df_norm.index)).fillna("").astype(str).to_numpy()
        sources = df_norm.get("source", pd.Series([""] * len(df_norm), index=df_norm.index)).fillna("").astype(str).to_numpy()

        left_pos, right_pos, left_idx_out, right_idx_out = self._pair_positions(pairs_df, idx_to_pos)
        if not left_pos:
            return _empty_matches()

        sims = self._pair_similarities(embeddings, left_pos, right_pos)

        records = []
        for lp, rp, li, rj, sim in zip(left_pos, right_pos, left_idx_out, right_idx_out, sims):
            if sim < cfg.threshold:
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
                    "similarity": float(sim),
                    "name_score": float(sim),  # compatibility with fuzzy matcher output
                    "match_type": "transformer_bge",
                }
            )

        if not records:
            return _empty_matches()

        return pd.DataFrame(records).sort_values("similarity", ascending=False).reset_index(drop=True)

    # --- Embedding helpers ---------------------------------------------

    def _build_text_inputs(
        self,
        df_norm: pd.DataFrame,
        name_col: str,
        desc_col: str,
        cfg: TransformerMatcherConfig,
    ) -> pd.Series:
        names = df_norm[name_col].fillna("").astype(str)
        desc = df_norm.get(desc_col, pd.Series([""] * len(df_norm), index=df_norm.index)).fillna("").astype(str)
        if cfg.max_desc_tokens and cfg.max_desc_tokens > 0:
            desc = desc.apply(lambda d: " ".join(d.split()[: cfg.max_desc_tokens]))
        # Weighting is handled by repeating tokens; simple and tokenizer-agnostic.
        name_part = names if cfg.name_weight >= 1.0 else names
        desc_part = desc
        if cfg.name_weight > 1.0:
            name_part = (name_part + " ") * int(cfg.name_weight)
        if cfg.desc_weight > 1.0:
            desc_part = (desc_part + " ") * int(cfg.desc_weight)
        return (name_part + " [SEP] " + desc_part).str.strip()

    def _get_embeddings(
        self,
        text_series: pd.Series,
        cfg: TransformerMatcherConfig,
    ) -> Tuple[np.ndarray, Dict[Any, int]]:
        cache_path = self._cache_path(text_series, cfg)
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            embeddings = data
        else:
            embeddings = self._encode(text_series, cfg)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, embeddings, allow_pickle=False)

        idx_to_pos = {idx: pos for pos, idx in enumerate(text_series.index)}
        return embeddings, idx_to_pos

    def _encode(self, text_series: pd.Series, cfg: TransformerMatcherConfig) -> np.ndarray:
        model = self._load_model(cfg)
        import torch

        tensor = model.encode(
            text_series.tolist(),
            batch_size=cfg.batch_size,
            device=cfg.device,
            convert_to_tensor=True,
            normalize_embeddings=False,
        )
        dtype = torch.float16 if cfg.use_fp16 else torch.float32
        embeddings = tensor.to(dtype).cpu().numpy()
        if cfg.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        return embeddings

    def _load_model(self, cfg: TransformerMatcherConfig):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(cfg.model_name, device=cfg.device)
        return self._model

    def _cache_path(self, text_series: pd.Series, cfg: TransformerMatcherConfig) -> Optional[Path]:
        if not cfg.cache_dir:
            return None
        hasher = hashlib.md5()
        hasher.update(cfg.model_name.encode("utf-8"))
        hasher.update(str(cfg.max_desc_tokens).encode("utf-8"))
        hasher.update(str(cfg.normalize_embeddings).encode("utf-8"))
        hasher.update(str(cfg.name_weight).encode("utf-8"))
        hasher.update(str(cfg.desc_weight).encode("utf-8"))
        # Stable hash of input texts + index to detect snapshot changes.
        hash_series = pd.util.hash_pandas_object(text_series, index=True)
        hasher.update(hash_series.values.tobytes())
        signature = hasher.hexdigest()
        model_slug = cfg.model_name.replace("/", "__")
        return Path(cfg.cache_dir) / model_slug / f"{signature}.npy"

    # --- Pair helpers ---------------------------------------------------

    def _pair_positions(
        self,
        pairs_df: pd.DataFrame,
        idx_to_pos: Dict[Any, int],
    ) -> Tuple[list[int], list[int], list[Any], list[Any]]:
        left_pos: list[int] = []
        right_pos: list[int] = []
        left_idx_out: list[Any] = []
        right_idx_out: list[Any] = []
        for left_idx, right_idx in zip(pairs_df["left_index"].to_numpy(), pairs_df["right_index"].to_numpy()):
            li = idx_to_pos.get(left_idx)
            rj = idx_to_pos.get(right_idx)
            if li is None or rj is None:
                continue
            left_pos.append(li)
            right_pos.append(rj)
            left_idx_out.append(left_idx)
            right_idx_out.append(right_idx)
        return left_pos, right_pos, left_idx_out, right_idx_out

    def _pair_similarities(self, embeddings: np.ndarray, left_pos: list[int], right_pos: list[int]) -> np.ndarray:
        left_vecs = embeddings[left_pos]
        right_vecs = embeddings[right_pos]
        return np.sum(left_vecs * right_vecs, axis=1)
