"""Fixed settings for every run. Changing any of these changes the cache keys."""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
REPORTS = os.path.join(ROOT, "reports")
FIGURES = os.path.join(REPORTS, "figures")
CACHE_DB = os.path.join(ROOT, "cache", "llm.sqlite")
EMB_CACHE = os.path.join(ROOT, "cache", "embeddings.npz")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELS = ["qwen3:8b", "llama3.1:8b", "gemma3:12b", "qwen3:30b-a3b"]
WRITER = "qwen3:8b"  # builds every memory store, so all readers see the same memory

SEED = 7
BASE_OPTIONS = {"temperature": 0, "seed": SEED, "num_ctx": 8192}
NUM_PREDICT = {"extract": 120, "overwrite": 16, "summary": 160, "answer": 80}

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

PILOT_USERS = [0, 1]  # used to develop and check the pipeline; excluded from results
USERS = list(range(2, 12))
K_DEFAULT = 5
K_LEVELS = [3, 5, 10]
OVERWRITE_NEIGHBOURS = 3  # existing entries shown to the writer per new fact
OVERWRITE_MAX_CANDIDATES = 8
RECENCY_DECAY = 0.95  # per session
RECENCY_WEIGHT = 1.0  # equal weight with normalised similarity, as in Park et al. (2023)
SUMMARY_WORDS = 80
BOOT = 2000


# This Ollama tag is the thinking-only Qwen3 release: its template always opens a <think> block and
# think: false does not stop the reasoning (it moves into the visible answer and the length cap cuts
# it off mid-sentence). It is therefore run with thinking on and a larger output cap; only the final
# answer is judged.
THINKING_ONLY = {"qwen3:30b-a3b"}
THINKING_NUM_PREDICT = 2048


def options(kind: str, model: str = "") -> dict:
    n = THINKING_NUM_PREDICT if model in THINKING_ONLY else NUM_PREDICT[kind]
    return {**BASE_OPTIONS, "num_predict": n}
