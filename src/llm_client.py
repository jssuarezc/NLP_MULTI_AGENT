"""LLM client for any OpenAI-compatible server (Ollama, OpenAI, ...).

Every call is cached on disk, keyed by (model, system, user, temperature, max_tokens, cache_tag).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from . import config


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    temperature: float
    max_tokens: int
    stop_reason: str = ""      
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class ResponseCache:
    """Append-only JSONL cache."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.open(encoding="utf-8"):
                try:
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec["response"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, key: str):
        return self._data.get(key)

    def put(self, key: str, response: dict) -> None:
        with self._lock:
            self._data[key] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": response}) + "\n")

    def __len__(self) -> int:
        return len(self._data)


def make_key(model, system, user, temperature, max_tokens, tag) -> str:
    blob = json.dumps([model, system, user, temperature, max_tokens, tag], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class LLMClient:
    def __init__(self, model: str | None = None, use_cache: bool = True):
        self.model = model or config.MODEL
        self.client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                             max_retries=config.MAX_RETRIES, timeout=600)
        self.cache = ResponseCache(config.CACHE_PATH) if use_cache else None
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cached_calls": 0, "input_tokens": 0, "output_tokens": 0}

    def generate(self, system: str, user: str, *, temperature: float, max_tokens: int,
                 model: str | None = None, cache_tag: str = "") -> LLMResponse:
        model = model or self.model
        key = make_key(model, system, user, temperature, max_tokens, cache_tag)
        hit = self.cache.get(key) if self.cache is not None else None
        if hit is not None:
            r = LLMResponse(**{**hit, "cached": True})
        else:
            extra = {"options": {"num_ctx": config.NUM_CTX}} if config.NUM_CTX else None
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens, extra_body=extra,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            latency = time.perf_counter() - t0
            choice = resp.choices[0]
            text = re.sub(r"<think>.*?</think>", "", choice.message.content or "", flags=re.S).strip()
            stop = "max_tokens" if choice.finish_reason == "length" else (choice.finish_reason or "")
            r = LLMResponse(text=text, input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens,
                            latency_s=latency, model=model, temperature=temperature, max_tokens=max_tokens,
                            stop_reason=stop)
            if self.cache is not None:
                self.cache.put(key, {**r.to_dict(), "cached": False})
        with self._lock:
            self.stats["calls"] += 1
            self.stats["cached_calls"] += int(r.cached)
            self.stats["input_tokens"] += r.input_tokens
            self.stats["output_tokens"] += r.output_tokens
        return r
