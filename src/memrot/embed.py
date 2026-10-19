"""Sentence embeddings with a small local model, cached by text."""

from __future__ import annotations

import os
import threading

import numpy as np

from . import config as C


class Embedder:
    def __init__(self, path: str | None = C.EMB_CACHE, model_name: str = C.EMBED_MODEL):
        self.path, self.model_name = path, model_name
        self._model = None
        self.lock = threading.Lock()
        self.vecs: dict[str, np.ndarray] = {}
        if path and os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            self.vecs = dict(zip(z["texts"].tolist(), z["vecs"]))
        self._dirty = False

    def _load(self):
        if self._model is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        with self.lock:
            todo = sorted({t for t in texts if t not in self.vecs})
            if todo:
                out = self._load().encode(todo, normalize_embeddings=True, batch_size=64)
                for t, v in zip(todo, out):
                    self.vecs[t] = v.astype(np.float32)
                self._dirty = True
            if not texts:
                return np.zeros((0, 384), dtype=np.float32)
            return np.stack([self.vecs[t] for t in texts])

    def query(self, text: str) -> np.ndarray:
        return self.encode([C.EMBED_QUERY_PREFIX + text])[0]

    def save(self):
        with self.lock:
            if not self.path or not self._dirty:
                return
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            texts = list(self.vecs)
            np.savez(self.path, texts=np.array(texts), vecs=np.stack([self.vecs[t] for t in texts]))
            self._dirty = False
