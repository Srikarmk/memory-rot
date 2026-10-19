import re
import zlib

import numpy as np
import pytest


class FakeEmbedder:
    """Bag-of-words hashing vectors, so tests need no model download."""

    def __init__(self, dim=512):
        self.dim = dim

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z]+", text.lower()):
            if w in {"the", "user", "a", "is", "in", "my", "i", "to", "of", "do", "what", "which"}:
                continue
            v[zlib.crc32(w.encode()) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def query(self, text):
        return self._vec(text)

    def save(self):
        pass


class FakeClient:
    """Answers by prompt type: extraction echoes each sentence as a fact, overwrite
    returns a scripted reply, summary appends the session text."""

    def __init__(self, overwrite_reply="NONE"):
        self.overwrite_reply = overwrite_reply
        self.calls = []

    def chat(self, model, messages, options, tag=""):
        text = messages[-1]["content"]
        self.calls.append(text)
        if text.startswith("Below is the user's message"):
            msg = text.split("Message:\n", 1)[1]
            sents = [s for s in re.split(r"(?<=[.!?])\s+", msg) if s and "?" not in s]
            out = "\n".join("The user said: " + s for s in sents) or "NONE"
        elif text.startswith("Existing memories"):
            out = self.overwrite_reply(text) if callable(self.overwrite_reply) else self.overwrite_reply
        elif text.startswith("Current summary"):
            prev = text.split("Current summary:\n", 1)[1].split("\n\nLatest", 1)[0]
            new = text.split("User: ", 1)[1].split("\n\nRewrite", 1)[0]
            out = (("" if prev == "(empty)" else prev + " ") + new).strip()
        else:
            out = "ok"
        return {"text": out, "prompt_tokens": len(text) // 4, "output_tokens": len(out) // 4,
                "seconds": 0.0, "cached": False}


@pytest.fixture
def emb():
    return FakeEmbedder()
