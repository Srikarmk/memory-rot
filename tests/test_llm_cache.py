import pytest

from memrot.llm import Cache, CacheMiss, Client, cache_key, request_body


def test_key_depends_on_everything():
    m = [{"role": "user", "content": "hi"}]
    o = {"temperature": 0, "seed": 7}
    k = cache_key("qwen3:8b", m, o)
    assert k == cache_key("qwen3:8b", m, dict(o))
    assert k != cache_key("llama3.1:8b", m, o)
    assert k != cache_key("qwen3:8b", m, {**o, "seed": 8})
    assert k != cache_key("qwen3:8b", [{"role": "user", "content": "hi!"}], o)
    assert k != cache_key("qwen3:8b", m, o, tag="rerun")


def test_think_flag():
    assert request_body("qwen3:8b", [], {})["think"] is False
    assert request_body("qwen3:30b-a3b", [], {})["think"] is True
    assert "think" not in request_body("gemma3:12b", [], {})


def test_cache_roundtrip_and_no_network_on_hit(tmp_path):
    cache = Cache(str(tmp_path / "c.sqlite"))
    m, o = [{"role": "user", "content": "x"}], {"temperature": 0}
    resp = {"text": "y", "raw": "y", "prompt_tokens": 3, "output_tokens": 1, "seconds": 0.1, "wall_seconds": 0.2}
    cache.put(cache_key("qwen3:8b", m, o), "qwen3:8b", resp)
    client = Client(cache, url="http://127.0.0.1:9", cache_only=True)
    r = client.chat("qwen3:8b", m, o)
    assert r["text"] == "y" and r["cached"] and client.stats["hits"] == 1
    with pytest.raises(CacheMiss):
        client.chat("qwen3:8b", m, o, tag="rerun")
    assert len(Cache(str(tmp_path / "c.sqlite"))) == 1
