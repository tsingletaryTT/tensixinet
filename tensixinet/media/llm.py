# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prompt an OpenAI-compatible endpoint (e.g. a local vLLM server).

The model name is read from the endpoint unless given, so the client works
against whatever the box happens to be serving. Standard library only, and the
opener is injectable so tests need no network.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Callable, Optional


class LLMError(RuntimeError):
    """The endpoint was unreachable or answered with something unusable."""


def _get_json(url: str, opener: Callable, timeout: float) -> dict:
    try:
        with opener(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (OSError, ValueError) as e:
        raise LLMError(str(e)) from e


def discover_model(endpoint: str, opener: Callable = urllib.request.urlopen, timeout: float = 5) -> str:
    data = _get_json(f"{endpoint}/v1/models", opener, timeout).get("data") or []
    if not data:
        raise LLMError(f"endpoint {endpoint} reports no models")
    return data[0]["id"]


def ask(
    prompt: str,
    endpoint: str,
    model: Optional[str] = None,
    opener: Callable = urllib.request.urlopen,
    max_tokens: int = 100,
    temperature: float = 0.7,
    timeout: float = 30,
) -> str:
    model = model or discover_model(endpoint, opener)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with opener(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (OSError, ValueError) as e:
        raise LLMError(str(e)) from e
    choices = data.get("choices")
    if not choices:
        raise LLMError(f"reply had no choices: {str(data)[:120]}")
    return choices[0].get("message", {}).get("content", "")
