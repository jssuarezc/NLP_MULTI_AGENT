"""Answer parsing, voting + tie-breaking, disagreement, answer-change analysis, cost, statistics.

"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from itertools import combinations

import numpy as np

from . import config

# Parsing
_FINAL_RE = re.compile(r"(?i:final\s+answer)\s*(?::|is)?[\s\*\(\[]*([A-H])(?![A-Za-z])")
_ANSWER_RE = re.compile(r"(?i:answer)\s*(?::|is)[\s\*\(\[]*([A-H])(?![A-Za-z])")
_PROPOSED_RE = re.compile(r"(?i:proposed\s+answer)\s*:?[\s\*\(\[]*([A-H])(?![A-Za-z])")
_BARE_RE = re.compile(r"\s*\(?([A-H])\)?[.\s]*")
_FIELD_RE = re.compile(r"(PROPOSED ANSWER|SUPPORTING ARGUMENT|CRITICISM OF ALTERNATIVE|FINAL ANSWER)\s*:", re.I)


def parse_answer(text: str, valid_letters) -> str | None:
    """Extract the final option letter. Prefers 'FINAL ANSWER', then 'Answer', then a bare letter."""
    valid = set(valid_letters)
    for pat in (_FINAL_RE, _ANSWER_RE):
        found = pat.findall(text or "")
        if found:
            return found[-1] if found[-1] in valid else None
    m = _BARE_RE.fullmatch(text or "")
    if m and m.group(1) in valid:
        return m.group(1)
    found = _PROPOSED_RE.findall(text or "")
    if found and found[-1] in valid:
        return found[-1]
    return None


def parse_structured(text: str) -> dict:
    """Split a structured response into its four fields (missing fields are absent)."""
    out: dict[str, str] = {}
    matches = list(_FIELD_RE.finditer(text or ""))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[m.group(1).lower().replace(" ", "_")] = text[m.end():end].strip()
    return out


# Voting
def majority_vote(answers, qid: str = "", rule: str = config.TIE_BREAK, seed: int = config.SEED) -> dict:
    """Plurality vote with a reproducible tie-break.

    None answers are ignored. If several options tie for the most votes:
      rule="hash"        -> pick tied option number (sha256(seed|qid) mod #tied) in alphabetical order.
                            Deterministic, independent of option position and of agent order.
      rule="first_agent" -> the tied option given by the lowest-indexed agent.
      rule="lowest"      -> alphabetically first tied option (biased toward A; for ablation only).
    All-None input yields answer None.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return {"answer": None, "tied": False, "counts": {}, "tied_options": []}
    counts = Counter(valid)
    top = max(counts.values())
    tied = sorted(k for k, v in counts.items() if v == top)
    if len(tied) == 1:
        choice = tied[0]
    elif rule == "hash":
        h = int(hashlib.sha256(f"{seed}|{qid}".encode()).hexdigest(), 16)
        choice = tied[h % len(tied)]
    elif rule == "first_agent":
        choice = next(a for a in valid if a in tied)
    elif rule == "lowest":
        choice = tied[0]
    else:
        raise ValueError(f"unknown tie-break rule {rule!r}")
    return {"answer": choice, "tied": len(tied) > 1, "counts": dict(counts), "tied_options": tied}


def is_correct(answer, gold) -> bool:
    return answer is not None and answer == gold


def pairwise_disagreement(answers) -> float:
    """Fraction of agent pairs giving different answers (None counts as its own answer)."""
    pairs = list(combinations(answers, 2))
    return sum(a != b for a, b in pairs) / len(pairs) if pairs else 0.0


def is_unanimous(answers) -> bool:
    return len(set(answers)) == 1


# Trajectories and per-round summaries
def padded_trajectory(rec: dict, n_rounds: int | None = None) -> list[list]:
    """answers_by_round, padded by repeating the last round (needed after early stopping)."""
    traj = [list(r) for r in rec["answers_by_round"]]
    target = n_rounds if n_rounds is not None else rec.get("max_rounds", len(traj) - 1) + 1
    while len(traj) < target:
        traj.append(list(traj[-1]))
    return traj


def round_summary(records: list[dict], rule: str = config.TIE_BREAK) -> list[dict]:
    """Per-round accuracy, disagreement and consensus for debate / self-revision records."""
    n_rounds = max(len(padded_trajectory(r)) for r in records)
    rows = []
    for r in range(n_rounds):
        maj_correct, agent_correct, dis, unan, wrong_cons, right_cons, none_ans, n_agent = ([] for _ in range(8))
        for rec in records:
            A = padded_trajectory(rec, n_rounds)[r]
            gold = rec["gold"]
            maj_correct.append(is_correct(majority_vote(A, rec["qid"], rule)["answer"], gold))
            agent_correct += [is_correct(a, gold) for a in A]
            none_ans += [a is None for a in A]
            dis.append(pairwise_disagreement(A))
            u = is_unanimous(A)
            unan.append(u)
            wrong_cons.append(u and not is_correct(A[0], gold))
            right_cons.append(u and is_correct(A[0], gold))
        rows.append({
            "round": r,
            "majority_acc": float(np.mean(maj_correct)),
            "mean_agent_acc": float(np.mean(agent_correct)),
            "pairwise_disagreement": float(np.mean(dis)),
            "unanimous_rate": float(np.mean(unan)),
            "unanimous_wrong_rate": float(np.mean(wrong_cons)),
            "unanimous_correct_rate": float(np.mean(right_cons)),
            "unparsed_rate": float(np.mean(none_ans)),
        })
    return rows


# Answer-change analysis (spec 7)
def _rate(n: int, d: int) -> float:
    return n / d if d else float("nan")


def answer_change_rows(records: list[dict], start: int = 0, end: int = -1,
                       rule: str = config.TIE_BREAK) -> list[dict]:
    c: Counter = Counter()
    for rec in records:
        traj = padded_trajectory(rec)
        A, B, gold, qid = traj[start], traj[end], rec["gold"], rec["qid"]
        n = len(A)
        # agent-level transitions
        for a, b in zip(A, B):
            c[("C" if a == gold else "I") + "->" + ("C" if b == gold else "I")] += 1
            c["changed"] += a != b
            c["agents"] += 1
        # question-level majority vote transitions
        ca = is_correct(majority_vote(A, qid, rule)["answer"], gold)
        cb = is_correct(majority_vote(B, qid, rule)["answer"], gold)
        c[("C" if ca else "I") + "->" + ("C" if cb else "I") + " vote"] += 1
        c["questions"] += 1
        # correct minority
        n_corr = sum(a == gold for a in A)
        wrong_counts = Counter(a for a in A if a is not None and a != gold)
        if n_corr >= 1 and n_corr < max(wrong_counts.values(), default=0):
            c["cm_q"] += 1
            c["cm_group_correct_after"] += cb
            c["cm_unanimous_correct_after"] += all(b == gold for b in B)
            for a, b in zip(A, B):
                if a == gold:
                    c["cm_correct_agents"] += 1
                    c["cm_correct_abandoned"] += b != gold
                else:
                    c["cm_wrong_agents"] += 1
                    c["cm_wrong_converted"] += b == gold
        # incorrect-majority pull (agent level)
        for i, (a, b) in enumerate(zip(A, B)):
            oc = Counter(x for j, x in enumerate(A) if j != i and x is not None)
            if a == gold:
                wrong = [(k, v) for k, v in oc.items() if k != gold]
                wk, wv = max(wrong, key=lambda kv: kv[1], default=(None, 0))
                if wv > (n - 1) / 2:
                    c["pull_den"] += 1
                    c["pull_to_that"] += b == wk
                    c["pull_to_any_wrong"] += b != gold
                else:
                    c["ctrl_den"] += 1
                    c["ctrl_to_wrong"] += b != gold
            elif oc.get(gold, 0) > (n - 1) / 2:
                c["push_den"] += 1
                c["push_to_correct"] += b == gold

    cc, ci, ic, ii = c["C->C"], c["C->I"], c["I->C"], c["I->I"]
    spec = [
        ("Agent correct -> incorrect", ci, cc + ci, "of agents correct at start"),
        ("Agent incorrect -> correct", ic, ic + ii, "of agents incorrect at start"),
        ("Agent changed answer", c["changed"], c["agents"], "of all agents"),
        ("Majority vote correct -> incorrect", c["C->I vote"], c["C->C vote"] + c["C->I vote"],
         "of questions with correct vote at start"),
        ("Majority vote incorrect -> correct", c["I->C vote"], c["I->C vote"] + c["I->I vote"],
         "of questions with incorrect vote at start"),
        ("Correct minority: group vote correct after", c["cm_group_correct_after"], c["cm_q"],
         "of questions with a correct minority"),
        ("Correct minority: unanimous correct after", c["cm_unanimous_correct_after"], c["cm_q"],
         "of questions with a correct minority"),
        ("Correct minority: wrong agents converted", c["cm_wrong_converted"], c["cm_wrong_agents"],
         "of wrong agents in those questions"),
        ("Correct minority: correct agents abandoned", c["cm_correct_abandoned"], c["cm_correct_agents"],
         "of correct agents in those questions"),
        ("Wrong majority pulls correct agent to that answer", c["pull_to_that"], c["pull_den"],
         "of correct agents facing a unified wrong majority"),
        ("Wrong majority: correct agent ends incorrect (any)", c["pull_to_any_wrong"], c["pull_den"],
         "of correct agents facing a unified wrong majority"),
        ("Control: correct agent (no wrong majority) ends incorrect", c["ctrl_to_wrong"], c["ctrl_den"],
         "of correct agents not facing a wrong majority"),
        ("Correct majority converts incorrect agent", c["push_to_correct"], c["push_den"],
         "of incorrect agents facing a correct majority"),
    ]
    return [{"metric": m, "num": int(n), "den": int(d), "rate": _rate(int(n), int(d)), "denominator": txt}
            for m, n, d, txt in spec]


def per_round_answer_changes(records: list[dict], rule: str = config.TIE_BREAK) -> dict[int, list[dict]]:
    n_rounds = max(len(padded_trajectory(r)) for r in records)
    return {r: answer_change_rows(records, r - 1, r, rule) for r in range(1, n_rounds)}


# Decision methods
def decision_comparison(records: list[dict], rule: str = config.TIE_BREAK) -> dict:
    """Majority vote vs LLM judge on the same final arguments; does the decision method change the outcome?"""
    rows = []
    for rec in records:
        if not rec.get("judge_final"):
            continue
        gold = rec["gold"]
        maj = majority_vote(padded_trajectory(rec)[-1], rec["qid"], rule)
        j = rec["judge_final"]["answer"]
        rows.append({"qid": rec["qid"], "majority": maj["answer"], "judge": j,
                     "majority_correct": is_correct(maj["answer"], gold), "judge_correct": is_correct(j, gold),
                     "differs": maj["answer"] != j, "majority_tied": maj["tied"]})
    n = len(rows)
    differ = [r for r in rows if r["differs"]]
    return {
        "n": n,
        "majority_acc": _rate(sum(r["majority_correct"] for r in rows), n),
        "judge_acc": _rate(sum(r["judge_correct"] for r in rows), n),
        "outcome_changed": len(differ),
        "outcome_changed_rate": _rate(len(differ), n),
        "judge_fixes_majority": sum(r["judge_correct"] and not r["majority_correct"] for r in rows),
        "judge_breaks_majority": sum(r["majority_correct"] and not r["judge_correct"] for r in rows),
        "both_wrong_but_differ": sum((not r["judge_correct"]) and (not r["majority_correct"]) and r["differs"] for r in rows),
        "rows": rows,
    }


# Cost
def usage(calls: list[dict], rounds=None, roles=None, agents=None) -> dict:
    """Tokens and latency over a filtered set of calls.

    latency_sum_s      : total model time if calls ran one after another
    latency_critical_s : wall-clock estimate with full parallelism inside each (role, round) group
    """
    sel = [c for c in calls
           if (rounds is None or c["round"] in rounds)
           and (roles is None or c["role"] in roles)
           and (agents is None or c["agent"] in agents)]
    groups: dict = {}
    for c in sel:
        groups.setdefault((c["role"], c["round"]), []).append(c["latency_s"])
    return {
        "calls": len(sel),
        "input_tokens": sum(c["input_tokens"] for c in sel),
        "output_tokens": sum(c["output_tokens"] for c in sel),
        "total_tokens": sum(c["input_tokens"] + c["output_tokens"] for c in sel),
        "latency_sum_s": sum(c["latency_s"] for c in sel),
        "latency_critical_s": sum(max(v) for v in groups.values()),
    }


def mean_usage(records: list[dict], **filters) -> dict:
    us = [usage(r["calls"], **filters) for r in records]
    return {k: float(np.mean([u[k] for u in us])) for k in us[0]} if us else {}


# Independent-sample baselines
def vote_at_k(rec: dict, k: int, rule: str = config.TIE_BREAK) -> dict:
    return majority_vote(rec["answers"][:k], rec["qid"], rule)


def k_for_budget(rec: dict, budget_tokens: float) -> int:
    """Largest number of independent samples (>=1) whose cumulative tokens fit the budget."""
    total, k = 0, 0
    for c in sorted(rec["calls"], key=lambda c: c["agent"]):
        total += c["input_tokens"] + c["output_tokens"]
        if total > budget_tokens and k >= 1:
            break
        k += 1
    return max(1, k)


# Statistics
def bootstrap_ci(correct, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(correct, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Paired exact (binomial) test on questions where exactly one method is right."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": a_only, "b_only": b_only, "delta_acc_b_minus_a": float(b.mean() - a.mean()), "p_value": p}


# Qualitative inspection helpers (spec 8)
def jaccard(a: str, b: str) -> float:
    ta, tb = set(re.findall(r"[a-z]+", a.lower())), set(re.findall(r"[a-z]+", b.lower()))
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


def _call_lookup(rec: dict) -> dict:
    return {(c["round"], c["agent"]): c for c in rec["calls"] if c["role"] == "agent"}


def find_examples(records: list[dict], kind: str, k: int = 10, seed: int = 0, repeat_threshold: float = 0.6) -> list[dict]:
    """Candidate examples for manual labelling.

    kind="useful_criticism"      : agent moved incorrect -> correct (shows the criticism it wrote)
    kind="unsupported_persuasion": agent moved correct -> incorrect toward an answer another agent held
    kind="repeated_argument"     : agent kept its answer and its text overlaps the previous turn
    """
    import random

    cands = []
    for rec in records:
        traj, gold, look = padded_trajectory(rec), rec["gold"], _call_lookup(rec)
        for r in range(1, len(traj)):
            for i, (a, b) in enumerate(zip(traj[r - 1], traj[r])):
                cur, prev = look.get((r, i)), look.get((r - 1, i))
                if cur is None or prev is None:
                    continue
                others_prev = [x for j, x in enumerate(traj[r - 1]) if j != i]
                hit = False
                if kind == "useful_criticism":
                    hit = a != gold and b == gold
                elif kind == "unsupported_persuasion":
                    hit = a == gold and b != gold and b in others_prev
                elif kind == "repeated_argument":
                    hit = a == b and jaccard(prev["text"], cur["text"]) >= repeat_threshold
                else:
                    raise ValueError(kind)
                if hit:
                    cands.append({
                        "kind": kind, "qid": rec["qid"], "agent": i, "round": r, "gold": gold,
                        "answers_before": traj[r - 1], "answers_after": traj[r],
                        "previous_text": prev["text"], "new_text": cur["text"],
                        "similarity": round(jaccard(prev["text"], cur["text"]), 3),
                        "criticism": parse_structured(cur["text"]).get("criticism_of_alternative"),
                    })
    random.Random(seed).shuffle(cands)
    return cands[:k]
