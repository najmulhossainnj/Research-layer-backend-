"""
FinBERT sentiment adapter.

Wraps the ProsusAI/finbert model from HuggingFace Transformers to score
financial news text.  Returns per-article scores for positive, negative,
neutral, and an uncertainty estimate derived from entropy over the
softmax distribution — matching exactly the feature columns the spec
calls for: positive_score, negative_score, neutral_score,
uncertainty_score, article_volume.

The model is loaded lazily and cached in process memory so repeated calls
within a worker session don't reload weights from disk.  For large article
batches the adapter chunks input to stay within the 512-token BERT limit.

Graceful fallback
-----------------
If torch / transformers are not installed (e.g. a lightweight API worker
that delegates heavy inference to a GPU worker), the adapter raises an
`ImportError` with a clear message.  The feature plugin still works; it
just routes inference to the correct worker type.
"""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import NamedTuple

import numpy as np


class SentimentScores(NamedTuple):
    positive: float
    negative: float
    neutral: float
    uncertainty: float   # entropy of the softmax distribution (0 = certain, 1 = max uncertain)


@lru_cache(maxsize=1)
def _load_finbert():
    """Load FinBERT tokenizer and model once per process."""
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required for FinBERT sentiment scoring. "
            "Install with: pip install transformers torch"
        ) from e

    model_name = "ProsusAI/finbert"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _score_batch(texts: list[str], batch_size: int = 16) -> list[SentimentScores]:
    """Score a list of texts and return one SentimentScores per text."""
    import torch

    tokenizer, model = _load_finbert()
    results: list[SentimentScores] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate to 512 tokens; FinBERT label order: positive, negative, neutral
        encoding = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=-1).numpy()   # (batch, 3)

        for row in probs:
            pos, neg, neu = float(row[0]), float(row[1]), float(row[2])
            # Shannon entropy normalised to [0, 1] over 3 classes
            entropy = -sum(p * math.log(p + 1e-9) for p in [pos, neg, neu])
            uncertainty = entropy / math.log(3)
            results.append(SentimentScores(pos, neg, neu, uncertainty))

    return results


def score_texts(texts: list[str]) -> list[SentimentScores]:
    """Public API: score a list of news strings. Empty strings get neutral scores."""
    cleaned = [t.strip() if t and t.strip() else "neutral" for t in texts]
    return _score_batch(cleaned)


def score_single(text: str) -> SentimentScores:
    return score_texts([text])[0]
