"""System instructions and prompt templates.
"""
from __future__ import annotations

from . import config

SYSTEM_SOLVER = (
    "You are an expert science teacher answering multiple-choice questions from a "
    "grade-school science exam. Reason carefully and commit to exactly one option letter."
)

SYSTEM_JUDGE = (
    "You are an impartial judge for multiple-choice science questions. You are shown "
    "anonymised solutions written by other solvers. You do not know which are correct. "
    "Agreement between solutions is not evidence of correctness; evaluate the scientific "
    "reasoning itself and commit to exactly one option letter."
)

# Single-agent templates
DIRECT_TEMPLATE = """{question_block}

Respond with only the letter of the correct option, in exactly this format:
Answer: <letter>"""

JUSTIFY_TEMPLATE = """{question_block}

First give a brief justification (at most {word_limit} words), then give your final choice on a last line in exactly this format:
Answer: <letter>"""

SELF_REVISE_TEMPLATE = """{question_block}

Your previous response:
\"\"\"
{own_previous}
\"\"\"

Re-examine your reasoning. First name one specific weakness or risk in it (or one specific strength you have verified), then revise your answer if warranted. Use at most {word_limit} words in total and end with a last line in exactly this format:
FINAL ANSWER: <letter>"""


# Debate templates (rounds >= 1)
DEBATE_FREE_TEMPLATE = """{question_block}

Your previous response:
\"\"\"
{own_previous}
\"\"\"

Other agents' previous responses:
{others_block}

Discuss freely. Before revising, point out one specific strength or weakness in another agent's argument. You may keep or change your answer. Use at most {word_limit} words in total and end with a last line in exactly this format:
FINAL ANSWER: <letter>"""

DEBATE_STRUCTURED_TEMPLATE = """{question_block}

Your previous response:
\"\"\"
{own_previous}
\"\"\"

Other agents' previous responses:
{others_block}

Respond using exactly these four fields, in this order, with at most {word_limit} words in total:
PROPOSED ANSWER: <letter you currently favour>
SUPPORTING ARGUMENT: <your strongest reasons for it>
CRITICISM OF ALTERNATIVE: <one specific strength or weakness in an argument for a different option; if all agents agree, name one specific weakness in the shared argument>
FINAL ANSWER: <letter, after taking the criticism into account>"""


# Judge template
JUDGE_TEMPLATE = """{question_block}

Below are {n} anonymised solutions in random order. Some may be wrong.

{solutions_block}

Decide which option is correct. Briefly assess which reasoning is soundest (at most {word_limit} words), then end with a last line in exactly this format:
FINAL ANSWER: <letter>"""

TEMPLATES = {
    "direct": DIRECT_TEMPLATE,
    "justify": JUSTIFY_TEMPLATE,
    "self_revise": SELF_REVISE_TEMPLATE,
    "debate_free": DEBATE_FREE_TEMPLATE,
    "debate_structured": DEBATE_STRUCTURED_TEMPLATE,
    "judge": JUDGE_TEMPLATE,
}



# Prompt Builders
def question_block(question: str, options: list, letters: list) -> str:
    lines = [f"Question: {question}", "Options:"]
    lines += [f"{l}. {o}" for l, o in zip(letters, options)]
    return "\n".join(lines)


def build_direct(qb: str) -> tuple[str, str]:
    return SYSTEM_SOLVER, DIRECT_TEMPLATE.format(question_block=qb)


def build_justify(qb: str) -> tuple[str, str]:
    return SYSTEM_SOLVER, JUSTIFY_TEMPLATE.format(question_block=qb, word_limit=config.WORD_LIMIT)


def build_self_revise(qb: str, own_previous: str) -> tuple[str, str]:
    return SYSTEM_SOLVER, SELF_REVISE_TEMPLATE.format(
        question_block=qb, own_previous=own_previous.strip(), word_limit=config.WORD_LIMIT)


def build_others_block(other_texts: list[str]) -> str:
    return "\n\n".join(f"[Other agent {i + 1}]\n{t.strip()}" for i, t in enumerate(other_texts))


def build_debate(qb: str, protocol: str, own_previous: str, other_texts: list[str]) -> tuple[str, str]:
    template = {"free": DEBATE_FREE_TEMPLATE, "structured": DEBATE_STRUCTURED_TEMPLATE}[protocol]
    return SYSTEM_SOLVER, template.format(
        question_block=qb, own_previous=own_previous.strip(),
        others_block=build_others_block(other_texts), word_limit=config.WORD_LIMIT)


def build_judge(qb: str, solution_texts: list[str]) -> tuple[str, str]:
    block = "\n\n".join(f"[Solution {i + 1}]\n{t.strip()}" for i, t in enumerate(solution_texts))
    return SYSTEM_JUDGE, JUDGE_TEMPLATE.format(
        question_block=qb, n=len(solution_texts), solutions_block=block, word_limit=config.WORD_LIMIT)
