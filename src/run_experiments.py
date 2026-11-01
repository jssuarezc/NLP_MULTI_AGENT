"""Headless runner for the full experiment suite (the notebooks do the same, interactively).

    python -m src.run_experiments --split dev
    python -m src.run_experiments --split test --stages baselines debate
    python -m src.run_experiments --split dev --limit 10   

Stages: data | baselines | debate
All transcripts go to results/transcripts/{split}_{name}.jsonl and are resumable.
"""
from __future__ import annotations

import argparse

from . import config, data_loader, prompts
from .debate_engine import (debate_name, run_dataset, run_debate, run_independent, run_self_revision,
                            run_single, transcript_path)
from .llm_client import LLMClient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--stages", nargs="+", default=["data", "baselines", "debate"],
                    choices=["data", "baselines", "debate"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    config.ensure_dirs()
    if "data" in args.stages:
        print("raw splits:", data_loader.download_arc())
        data_loader.build_samples()
    qs = data_loader.load_sampled(args.split)[: args.limit]
    client = LLMClient()
    print(f"model={config.MODEL!r}  server={config.BASE_URL}  num_ctx={config.NUM_CTX or 'default'}  workers={config.MAX_WORKERS}")
    try:  
        r = client.generate("You are a helpful assistant.", "Reply with the single word READY.",
                            temperature=0.0, max_tokens=10, cache_tag="ping")
        print(f"test call OK: {r.text!r} ({r.latency_s:.1f}s)")
    except Exception as exc: 
        raise SystemExit(f"Test call failed: {exc}\nCheck LLM_MODEL (run `ollama list`), LLM_BASE_URL and that Ollama is running.")
    W = args.workers
    tp = lambda name: transcript_path(args.split, name)  # noqa: E731

    if "baselines" in args.stages:
        run_dataset(lambda q: run_single(client, q, "direct"), qs, tp("single_direct"), W)
        run_dataset(lambda q: run_single(client, q, "justify"), qs, tp("single_justify"), W)
        run_dataset(lambda q: run_independent(client, q, config.MAX_INDEP_SAMPLES, "justify"), qs,
                    tp("independent_justify"), W)
        run_dataset(lambda q: run_independent(client, q, config.N_AGENTS, "direct"), qs,
                    tp("independent_direct"), W)
        run_dataset(lambda q: run_self_revision(client, q, config.MAX_ROUNDS), qs, tp("selfrevise"), W)
    if "debate" in args.stages:
        for protocol in ("free", "structured"):
            for icl in (False, True):     
                run_dataset(lambda q, p=protocol, i=icl: run_debate(client, q, p, icl=i), qs,
                            tp(debate_name(protocol, icl)), W,
                            expect={"icl": icl, "icl_version": prompts.ICL_VERSION if icl else None})
    print("client stats:", client.stats)


if __name__ == "__main__":
    main()