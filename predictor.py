"""Predict exactly one masked vocabulary token using BERT."""
from __future__ import annotations

from time import perf_counter
from typing import Any

MODEL_ID = "google-bert/bert-base-uncased"


def validate_input(text: str, top_k: int) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty sentence.")
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("The number of suggestions must be an integer from 1 to 10.")
    text = text.strip()
    if text.count("[MASK]") != 1:
        raise ValueError("Use exactly one [MASK] marker.")
    return text


class MaskPredictor:
    def __init__(self, *, device: int = -1, revision: str | None = None):
        self.device = device
        self.revision = revision
        self._pipeline: Any = None

    def load(self):
        if self._pipeline is None:
            from transformers import pipeline

            kwargs = {"revision": self.revision} if self.revision else {}
            self._pipeline = pipeline("fill-mask", model=MODEL_ID, device=self.device, **kwargs)

    def predict(self, text: str, top_k: int = 5) -> dict:
        text = validate_input(text, top_k)
        self.load()
        tokenizer = self._pipeline.tokenizer
        ids = tokenizer(text)["input_ids"]
        if ids.count(tokenizer.mask_token_id) != 1:
            raise ValueError("Exactly one encoded mask position is required.")
        if len(ids) > tokenizer.model_max_length:
            raise ValueError("The sentence exceeds the model token limit.")
        started = perf_counter()
        predictions = self._pipeline(text, top_k=top_k)
        return {
            "input": text, "top_k": top_k,
            "predictions": [{"token": int(p["token"]), "token_str": p["token_str"],
                             "score": float(p["score"]), "sequence": p["sequence"]}
                            for p in predictions],
            "model": MODEL_ID,
            "revision": self.revision or getattr(self._pipeline.model.config, "_commit_hash", None),
            "inference_seconds": round(perf_counter() - started, 4),
        }
