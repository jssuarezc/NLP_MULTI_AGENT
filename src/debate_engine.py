"""Experiment runners: single agent, independent samples, self-revision, synchronous debate, LLM judge.

All runners take a data_loader.Question and return a JSON-serialisable record containing every
prompt and response (the complete transcript). Gold answers are copied into records for later
analysis but are never passed to prompt builders.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from . import config, metrics, prompts
from .data_loader import Question
from .llm_client import LLMClient


# Low-level helpers
def _qb(q: Question) -> str:
    return prompts.question_block(q.question, q.options, q.letters)


def _call(client: LLMClient, q: Question, system: str, user: str, spec_key: str, *,
          role: str, agent: int, rnd: int, cache_tag: str, model: str | None = None) -> dict:
    spec = config.GEN[spec_key]
    r = client.generate(system, user, temperature=spec.temperature, max_tokens=spec.max_tokens,
                        model=model, cache_tag=cache_tag)
    answer = metrics.parse_answer(r.text, q.letters)
    return {"agent": agent, "round": rnd, "role": role, "answer": answer, "parse_ok": answer is not None,
            "system": system, "prompt": user, **r.to_dict()}


def _parallel(fns: list[Callable[[], dict]]) -> list[dict]:
    if len(fns) == 1:
        return [fns[0]()]
    with ThreadPoolExecutor(min(len(fns), config.MAX_WORKERS)) as ex:
        return list(ex.map(lambda f: f(), fns))


def _base(q: Question, **extra) -> dict:
    return {"qid": q.id, "split": q.split, "gold": q.answer, "model": config.MODEL, **extra}


# Single agent / independent samples / self-revision
def run_single(client: LLMClient, q: Question, mode: str = "justify") -> dict:
    """mode: "direct" (letter only) or "justify" (short justification + letter), temperature 0."""
    build = {"direct": prompts.build_direct, "justify": prompts.build_justify}[mode]
    system, user = build(_qb(q))
    call = _call(client, q, system, user, mode, role="agent", agent=0, rnd=0, cache_tag=f"{q.id}|single-{mode}")
    return _base(q, mode=mode, answer=call["answer"], calls=[call])


def sample_independent(client: LLMClient, q: Question, k: int, mode: str = "justify") -> dict:
    build = {"direct": prompts.build_direct, "justify": prompts.build_justify}[mode]
    system, user = build(_qb(q))
    return _call(client, q, system, user, f"sample_{mode}", role="agent", agent=k, rnd=0,
                 cache_tag=f"{q.id}|{mode}|s{k}")


def run_independent(client: LLMClient, q: Question, n: int = config.N_AGENTS, mode: str = "justify") -> dict:
    """n independent calls that never see each other's output, at T > 0."""
    calls = _parallel([lambda k=k: sample_independent(client, q, k, mode) for k in range(n)])
    answers = [c["answer"] for c in calls]
    return _base(q, mode=mode, n=n, answers=answers, calls=calls,
                 vote=metrics.majority_vote(answers, q.id))


def run_self_revision(client: LLMClient, q: Question, rounds: int = config.MAX_ROUNDS) -> dict:
    """One agent revises its own answer for several rounds, seeing only its own previous response."""
    qb = _qb(q)
    calls = [sample_independent(client, q, 0, "justify")]
    answers = [[calls[0]["answer"]]]
    for r in range(1, rounds + 1):
        system, user = prompts.build_self_revise(qb, calls[-1]["text"])
        c = _call(client, q, system, user, "self_revise", role="agent", agent=0, rnd=r,
                  cache_tag=f"{q.id}|selfrev|r{r}")
        calls.append(c)
        answers.append([c["answer"]])
    return _base(q, mode="self_revise", max_rounds=rounds, answers_by_round=answers, calls=calls)


# Judge
def judge_decide(client: LLMClient, q: Question, texts: list[str], role: str, rnd: int) -> dict:
    """LLM judge over anonymised arguments. Order is shuffled reproducibly; gold is never shown."""
    seed = int(hashlib.sha256(f"{config.SEED}|{q.id}|{role}".encode()).hexdigest(), 16)
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    system, user = prompts.build_judge(_qb(q), [texts[i] for i in order])
    call = _call(client, q, system, user, "judge", role=role, agent=0, rnd=rnd,
                 cache_tag=f"{q.id}|{role}", model=config.JUDGE_MODEL)
    call["order"] = order 
    return call


# Debate
def run_debate(client: LLMClient, q: Question, protocol: str = "structured",
               n_agents: int = config.N_AGENTS, rounds: int = config.MAX_ROUNDS,
               early_stop: bool = False, judge_final: bool = True, judge_round0: bool = True,
               icl: bool = False) -> dict:
    """Synchronous debate between n_agents, with optional ICL examples and LLM judge."""
    assert protocol in ("free", "structured")
    assert isinstance(icl, bool), "icl must be True or False"
    qb = _qb(q)
    r0 = _parallel([lambda k=k: sample_independent(client, q, k, "justify") for k in range(n_agents)])
    calls = list(r0)
    prev_texts = [c["text"] for c in r0]
    answers_by_round = [[c["answer"] for c in r0]]
    stopped_after = None

    for r in range(1, rounds + 1):
        last = answers_by_round[-1]
        if early_stop and metrics.is_unanimous(last) and last[0] is not None:
            stopped_after = r - 1
            break
        prompts_r = []
        for i in range(n_agents):
            others = [prev_texts[j] for j in range(n_agents) if j != i]
            prompts_r.append(prompts.build_debate(qb, protocol, prev_texts[i], others, icl=icl))
        tag_suffix = "-icl" if icl else ""
        cur = _parallel([
            (lambda i=i, s=s, u=u: _call(client, q, s, u, "debate", role="agent", agent=i, rnd=r,
                                         cache_tag=f"{q.id}|debate-{protocol}{tag_suffix}|r{r}|a{i}"))
            for i, (s, u) in enumerate(prompts_r)])
        calls += cur
        prev_texts = [c["text"] for c in cur]
        answers_by_round.append([c["answer"] for c in cur])

    rec = _base(q, mode=debate_name(protocol, icl), protocol=protocol, icl=icl,
               icl_version=prompts.ICL_VERSION if icl else None,
               n_agents=n_agents, max_rounds=rounds,
               early_stopped_after_round=stopped_after, answers_by_round=answers_by_round)

    if judge_round0:
        j0 = judge_decide(client, q, [c["text"] for c in r0], "judge_round0", 0)
        calls.append(j0)
        rec["judge_round0"] = {"answer": j0["answer"], "order": j0["order"], "text": j0["text"]}
    if judge_final:
        jf = judge_decide(client, q, prev_texts, "judge_final", len(answers_by_round) - 1)
        calls.append(jf)
        rec["judge_final"] = {"answer": jf["answer"], "order": jf["order"], "text": jf["text"]}
    rec["majority_final"] = metrics.majority_vote(answers_by_round[-1], q.id)
    rec["calls"] = calls
    return rec


# Dataset runner
def debate_name(protocol: str, icl: bool = False) -> str:
    """Returns a unique name for a debate variant, used in transcript filenames. protocol: "free" | "structured"."""
    return f"debate_{protocol}{'_icl' if icl else ''}"


def transcript_path(split: str, name: str) -> Path:
    return config.TRANSCRIPT_DIR / f"{split}_{name}.jsonl"


def load_records(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_dataset(fn: Callable[[Question], dict], questions: list[Question], out_path: Path,
                workers: int | None = None, resume: bool = True, desc: str = "",
                expect: dict | None = None) -> list[dict]:

    config.ensure_dirs()
    out_path = Path(out_path)
    done = {r["qid"]: r for r in load_records(out_path)} if resume else {}
    if expect:
        stale = [qid for qid, r in done.items() if any((r.get(k) or None) != (v or None) for k, v in expect.items())]
        if stale:
            raise ValueError(f"{out_path.name}: {len(stale)} stored record(s) do not match {expect} ")
    if not resume and out_path.exists():
        out_path.unlink()
    todo = [q for q in questions if q.id not in done]
    lock, errors = threading.Lock(), []
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(todo), desc=desc or out_path.stem)
    except ImportError: 
        bar = None

    with ThreadPoolExecutor(workers or config.MAX_WORKERS) as ex, out_path.open("a", encoding="utf-8") as f:
        futs = {ex.submit(fn, q): q for q in todo}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc: 
                errors.append((q.id, repr(exc)))
                continue
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            done[q.id] = rec
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    if errors:
        print(f"[run_dataset] {len(errors)} question(s) failed (re-run to retry), e.g. {errors[:2]}")
    return [done[q.id] for q in questions if q.id in done]


# Readable transcript
def render_transcript(rec: dict, include_prompts: bool = False) -> str:
    lines = [f"# {rec['qid']}  (gold: {rec['gold']}, mode: {rec.get('mode')})", ""]
    for c in rec["calls"]:
        lines.append(f"## round {c['round']} · {c['role']} · agent {c['agent']} → answer {c['answer']}"
                     f"  ({c['input_tokens']}+{c['output_tokens']} tok, {c['latency_s']:.2f}s)")
        if include_prompts:
            lines += ["", "```", c["prompt"], "```"]
        lines += ["", c["text"].strip(), ""]
    return "\n".join(lines)