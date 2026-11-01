# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Put real load on the chips that serve a model, in bursts, so a chip loads, then falls back to rest.

tensixinet's evict feature fires when a chip that has seen a load falls back to rest. A model that is served but not asked anything
is at rest, so a recording needs prompts. This sends bursts of long completions at an OpenAI-compatible endpoint, a few at a time,
then waits. The prompts are long and the answers short (see PROMPT). Plain HTTP only; it opens no device.

    python3 demo/load_prompts.py --endpoint http://127.0.0.1:8011 --model Qwen/Qwen3-Coder-Next --burst 25 --rest 30 --cycles 3
"""
import argparse
import json
import threading
import time
import urllib.request

# A long prompt with a very short answer keeps the chips in prefill, which draws 120 to 180 W on a Blackhole with this model resident, against
# 35 to 45 W at rest. Long answers (decode) only reach about 50 to 57 W, which never clears the evict rule's 1.5 times rest.
# `--decode-streams` adds streams that ask for a long answer to a short question. They keep the chips decoding, which draws far less than prefill
# but runs for as long as the answer does: the model visibly "thinking" while the prefill streams come and go.
THINK_PROMPT = "Write a complete Python module that implements a thread-safe LRU cache with a TTL, type hints, docstrings and unit tests."
PROMPT = ("The quick brown fox jumps over the lazy dog while the hash table resizes. " * 1500) + "\nSummarise the text above in detail."


def ask(endpoint: str, model: str, max_tokens: int, timeout: float, prompt: str = PROMPT) -> None:
    body = json.dumps({"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", body, {"content-type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=timeout).read()
    except OSError:
        pass  # a timeout or a refusal ends this request; the burst just carries on


def burst(endpoint: str, model: str, seconds: float, streams: int, max_tokens: int, decode_streams: int = 0, decode_tokens: int = 400) -> None:
    """`streams` workers each send completions back to back until `seconds` have passed; `decode_streams` more ask for long answers."""
    end = time.monotonic() + seconds

    def worker(prompt: str, tokens: int) -> None:
        while time.monotonic() < end:
            ask(endpoint, model, tokens, timeout=max(5.0, end - time.monotonic() + 30), prompt=prompt)

    threads = [threading.Thread(target=worker, args=(PROMPT, max_tokens)) for _ in range(streams)]
    threads += [threading.Thread(target=worker, args=(THINK_PROMPT, decode_tokens)) for _ in range(decode_streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--endpoint", default="http://127.0.0.1:8011")
    p.add_argument("--model", default="Qwen/Qwen3-Coder-Next")
    p.add_argument("--burst", type=float, default=25, help="seconds of load per cycle")
    p.add_argument("--rest", type=float, default=30, help="seconds of rest per cycle")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--streams", type=int, default=16, help="requests in flight during a burst")
    p.add_argument("--max-tokens", type=int, default=4, help="answer length; short keeps the chips in prefill")
    p.add_argument("--decode-streams", type=int, default=0, help="extra streams that ask for a long answer, so the model is decoding as well as reading")
    p.add_argument("--lead", type=float, default=0, help="seconds to wait before the first burst")
    a = p.parse_args()
    time.sleep(a.lead)
    for i in range(a.cycles):
        print(f"load {i + 1}/{a.cycles}: {a.streams} streams for {a.burst:g} s", flush=True)
        burst(a.endpoint, a.model, a.burst, a.streams, a.max_tokens, a.decode_streams)
        print(f"rest {a.rest:g} s", flush=True)
        time.sleep(a.rest)


if __name__ == "__main__":
    main()
