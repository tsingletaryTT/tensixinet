"""LLM prompting: model discovered from the endpoint, failures reported not hidden."""
import io
import json

import pytest

from tensixinet.media.llm import LLMError, ask, discover_model


class FakeResponse(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def opener_for(routes, seen=None):
    def opener(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if seen is not None:
            seen.append(req)
        body = routes[url.split("://", 1)[1].split("/", 1)[1]]
        return FakeResponse(json.dumps(body).encode())
    return opener


MODELS = {"v1/models": {"data": [{"id": "acme/served-model-7b"}, {"id": "other"}]}}


def test_discover_model_picks_the_first_model_the_endpoint_serves():
    assert discover_model("http://h:1", opener_for(MODELS)) == "acme/served-model-7b"


def test_discover_model_errors_when_none_served():
    with pytest.raises(LLMError, match="no models"):
        discover_model("http://h:1", opener_for({"v1/models": {"data": []}}))


def test_ask_sends_the_discovered_model_and_prompt_and_returns_the_reply():
    seen = []
    routes = {**MODELS, "v1/chat/completions": {"choices": [{"message": {"content": "hello"}}]}}
    reply = ask("think about rhythm", "http://h:1", opener=opener_for(routes, seen), max_tokens=50)
    assert reply == "hello"
    post = seen[-1]
    body = json.loads(post.data)
    assert body["model"] == "acme/served-model-7b"
    assert body["messages"] == [{"role": "user", "content": "think about rhythm"}]
    assert body["max_tokens"] == 50
    assert post.get_header("Content-type") == "application/json"


def test_ask_uses_an_explicit_model_without_asking_the_endpoint():
    seen = []
    routes = {"v1/chat/completions": {"choices": [{"message": {"content": "x"}}]}}
    ask("p", "http://h:1", model="mine", opener=opener_for(routes, seen))
    assert len(seen) == 1 and json.loads(seen[0].data)["model"] == "mine"


def test_ask_wraps_network_failures_in_llmerror():
    def boom(req, timeout=None):
        raise OSError("connection refused")
    with pytest.raises(LLMError, match="connection refused"):
        ask("p", "http://h:1", model="m", opener=boom)


def test_ask_rejects_a_reply_with_no_choices():
    routes = {"v1/chat/completions": {"error": "overloaded"}}
    with pytest.raises(LLMError, match="no choices"):
        ask("p", "http://h:1", model="m", opener=opener_for(routes))
