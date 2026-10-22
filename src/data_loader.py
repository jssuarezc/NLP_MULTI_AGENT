"""Download, standardise and sample ARC-Challenge; light dataset-analysis helpers.

Options are preserved verbatim and in their original order. They are only
re-lettered A, B, C, ... (ARC uses "1".."4" labels for a few questions); the
original labels are kept on each Question for traceability.
"""
from __future__ import annotations

import io
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config

LETTERS = "ABCDEFGH"


@dataclass
class Question:
    id: str
    split: str                 # "dev" or "test" once sampled; raw split name before
    question: str
    options: list              # option texts, original order
    letters: list              # ["A", "B", ...] aligned with options
    answer: str                # gold letter (NEVER shown to any agent or judge)
    orig_labels: list
    orig_answer_key: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(**d)

    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def answer_text(self) -> str:
        return self.options[self.letters.index(self.answer)]


# Download / standardise
_PARQUET_URL = ("https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/"
                "ARC-Challenge/{split}-00000-of-00001.parquet")


def standardize(row: dict, split: str) -> Question:
    labels = [str(x) for x in row["choices"]["label"]]
    texts = [str(x) for x in row["choices"]["text"]]
    key = str(row["answerKey"]).strip()
    if key not in labels:
        raise ValueError(f"answerKey {key!r} not in labels {labels} for {row.get('id')}")
    letters = list(LETTERS[: len(texts)])
    return Question(
        id=str(row["id"]),
        split=split,
        question=str(row["question"]).strip(),
        options=texts,
        letters=letters,
        answer=letters[labels.index(key)],
        orig_labels=labels,
        orig_answer_key=key,
    )


def _fetch_split(split: str) -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(config.HF_DATASET, config.HF_CONFIG, split=split)
        return [dict(r) for r in ds]
    except Exception as exc:  
        print(f"[data_loader] `datasets` route failed ({type(exc).__name__}); trying parquet URL")
        import pandas as pd
        import requests

        resp = requests.get(_PARQUET_URL.format(split=split), timeout=180)
        resp.raise_for_status()
        return pd.read_parquet(io.BytesIO(resp.content)).to_dict("records")


def download_arc(force: bool = False) -> dict:
    """Download all three official splits into data/raw/{split}.jsonl."""
    config.ensure_dirs()
    counts = {}
    for split in ("train", "validation", "test"):
        out = config.RAW_DIR / f"{split}.jsonl"
        if out.exists() and not force:
            counts[split] = sum(1 for _ in out.open(encoding="utf-8"))
            continue
        rows = _fetch_split(split)
        qs = [standardize(r, split) for r in rows]
        with out.open("w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q.to_dict()) + "\n")
        counts[split] = len(qs)
    return counts


def _read_jsonl(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        return [Question.from_dict(json.loads(line)) for line in f if line.strip()]


def load_raw(split: str) -> list[Question]:
    return _read_jsonl(config.RAW_DIR / f"{split}.jsonl")


# Sampling
def sample_questions(source_split: str, n: int, seed: int = config.SEED) -> list[Question]:
    pool = sorted(load_raw(source_split), key=lambda q: q.id)
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def build_samples(force: bool = False) -> dict:
    """Write data/sampled/dev.jsonl (validation) and test.jsonl (test)."""
    config.ensure_dirs()
    specs = {"dev": (config.DEV_SOURCE_SPLIT, config.N_DEV), "test": (config.TEST_SOURCE_SPLIT, config.N_TEST)}
    manifest = {}
    for name, (src, n) in specs.items():
        out = config.SAMPLED_DIR / f"{name}.jsonl"
        if out.exists() and not force:
            qs = load_sampled(name)
        else:
            qs = sample_questions(src, n)
            for q in qs:
                q.split = name
            with out.open("w", encoding="utf-8") as f:
                for q in qs:
                    f.write(json.dumps(q.to_dict()) + "\n")
        manifest[name] = {"source_split": src, "n": len(qs), "seed": config.SEED, "ids": [q.id for q in qs]}
    (config.SAMPLED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load_sampled(name: str) -> list[Question]:
    """name in {"dev", "test"}."""
    return _read_jsonl(config.SAMPLED_DIR / f"{name}.jsonl")


# Dataset analysis helpers (used by notebook 01)
import nltk
from nltk import FreqDist, bigrams
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

_tokenizer = RegexpTokenizer(r"[A-Za-z]+(?:'[a-z]+)?")   # words only;
_stopwords_cache: set = set()


def get_stopwords() -> set:
    """NLTK English stopword list (downloaded once on first use)."""
    if not _stopwords_cache:
        try:
            stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        _stopwords_cache.update(stopwords.words("english"))
    return _stopwords_cache


def tokenize_words(text: str) -> list[str]:
    return [t.lower() for t in _tokenizer.tokenize(text)]


def content_words(text: str) -> list[str]:
    """Lower-cased word tokens without stopwords and without 1-2 letter tokens."""
    sw = get_stopwords()
    return [w for w in tokenize_words(text) if w not in sw and len(w) > 2]


def word_freqdist(questions: list[Question]) -> FreqDist:
    """FreqDist of content words in the question stems (stopwords removed)."""
    return FreqDist(w for q in questions for w in content_words(q.question))


def top_bigrams(questions: list[Question], k: int = 20) -> list:
    """Most frequent word bigrams in question stems; stopwords are kept so phrases like 'which of' appear."""
    c = FreqDist(b for q in questions for b in bigrams(tokenize_words(q.question)))
    return c.most_common(k)


def wh_word(q: Question) -> str:
    """First wh-word / question opener of the stem ('which', 'what', 'how', ...), else 'other'."""
    for w in tokenize_words(q.question):
        if w in {"which", "what", "why", "how", "when", "where", "who"}:
            return w
    return "other"


if __name__ == "__main__":
    print("Full splits:", download_arc())
    print({k: (v["source_split"], v["n"]) for k, v in build_samples().items()})
