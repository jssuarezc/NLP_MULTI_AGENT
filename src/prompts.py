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

# In-context learning (ICL) examples for the debate protocols (used when build_debate(..., icl=True)).
#
#   1. WRONG own answer meets a specific, correct argument  -> switch, and say what was wrong with the old reasoning
#   2. CORRECT own answer meets two confident but wrong agents -> keep it by rebutting their claim
#      (+ a POOR response to the same situation: giving in to the majority with no argument)
#   3. ALL agents agree and are correct (the common case, ~94% of real questions) -> keep it, name one specific gap

ICL_SCENARIOS = [
    {
        "title": "good revision: switching to the better-supported answer",
        "question": "Which observation would best support the claim that a rock sample formed from cooled lava?",
        "options": ["It contains fossils of small sea animals.",
                    "It is made of rounded sand grains pressed together.",
                    "It contains interlocking crystals and no layers.",
                    "It is made of thin layers of different colors."],
        "own_previous": ("Lava flows pile up in sheets over many eruptions, so a rock with thin layers of "
                         "different colors probably formed from lava.\n\nAnswer: D"),
        "others": [
            ("Cooled lava forms igneous rock. As molten rock cools, its minerals crystallize and lock together, "
             "so the sample should show interlocking crystals. Layers suggest sediment, and fossils or rounded "
             "grains point to sedimentary rock.\n\nAnswer: C"),
            ("Igneous rock forms when magma or lava cools and hardens, which gives interlocking crystals rather "
             "than layers of settled sediment. Fossils and pressed sand grains show a sedimentary origin.\n\nAnswer: C"),
        ],
        "good_free": (
            "Other agent 1 gives the stronger argument: cooled lava crystallizes into interlocking mineral grains, "
            "while layers come from sediment settling over time. My earlier claim that layered rock comes from "
            "layered lava flows ignored how lava cools. The sample with interlocking crystals and no layers fits "
            "cooled lava best.\n\nFINAL ANSWER: C"),
        "good_structured": (
            "PROPOSED ANSWER: C\n"
            "SUPPORTING ARGUMENT: Lava that cools and hardens forms igneous rock, so the sample should show "
            "interlocking crystals and no layers.\n"
            "CRITICISM OF ALTERNATIVE: My earlier argument for D was weak: layered rock usually comes from sediment "
            "settling over time, and Other agent 1's point about crystallization explains cooled lava correctly.\n"
            "FINAL ANSWER: C"),
    },
    {
        "title": "good revision: keeping a correct answer against confident but wrong agents",
        "question": ("In the northern hemisphere, plants next to the south wall of a barn grow taller in early "
                     "spring than plants next to the north wall. What is the most likely reason?"),
        "options": ["The south wall gets more direct sunlight, so the soil is warmer.",
                    "The south side of the barn has more oxygen in the air.",
                    "The north side of the barn receives more rain.",
                    "Plants grow toward the south because of Earth's magnetic field."],
        "own_previous": ("In the northern hemisphere the sun stays in the southern part of the sky, so the south "
                         "wall gets more direct sunlight. The warmer soil and extra light help seeds grow faster "
                         "in early spring.\n\nAnswer: A"),
        "others": [
            ("Warm air carries more oxygen, and the south wall is warmer, so the air there has more oxygen for the "
             "plants to use. More oxygen means faster growth.\n\nAnswer: B"),
            ("Plants need oxygen for respiration, and the sunny south side has more oxygen in the air than the "
             "shaded north side, so plants there grow taller.\n\nAnswer: B"),
        ],
        "good_free": (
            "Other agents 1 and 2 claim the south side has more oxygen, but air is about 21% oxygen on both sides "
            "of the barn, and warm air does not carry more of it. The difference is better explained by the south "
            "wall's extra direct sunlight, which warms the soil and gives seedlings more light in early spring. "
            "I keep my answer.\n\nFINAL ANSWER: A"),
        "good_structured": (
            "PROPOSED ANSWER: A\n"
            "SUPPORTING ARGUMENT: The sun stays in the southern sky in the northern hemisphere, so the south wall "
            "gets more direct sunlight, which warms the soil and speeds early growth.\n"
            "CRITICISM OF ALTERNATIVE: Other agents 1 and 2 say warm air carries more oxygen, but air is about 21% "
            "oxygen on both sides of the barn, so oxygen cannot explain the difference.\n"
            "FINAL ANSWER: A"),
        "poor_free": ("Two other agents chose B, so they are probably right. I will go with the majority.\n\n"
                      "FINAL ANSWER: B"),
        "poor_structured": ("PROPOSED ANSWER: B\n"
                            "SUPPORTING ARGUMENT: The other agents agree on B.\n"
                            "CRITICISM OF ALTERNATIVE: Option A might be less likely.\n"
                            "FINAL ANSWER: B"),
        "poor_reason": ("It gave up a correct answer only because two agents agreed, gave no scientific reason, and "
                        "offered no specific criticism. Agreement between agents is not evidence."),
    },
    {
        "title": "good revision: all agents agree, so check the shared argument for a gap",
        "question": ("A student measures the mass of an ice cube, lets it melt completely, and then measures the "
                     "mass of the water. What will the student most likely find?"),
        "options": ["The water has less mass because it is a liquid.",
                    "The water has the same mass as the ice cube.",
                    "The water has more mass because it is denser.",
                    "The mass cannot be measured after melting."],
        "own_previous": ("Melting only changes the state of the water from solid to liquid. No matter is added or "
                         "removed, so the mass stays the same.\n\nAnswer: B"),
        "others": [
            ("Melting is a physical change. The same particles are present before and after, so the total mass is "
             "conserved.\n\nAnswer: B"),
            ("Mass is conserved during a change of state, so the water must have the same mass as the ice cube. "
             "The volume may change, but not the mass.\n\nAnswer: B"),
        ],
        "good_free": (
            "All three of us chose B and the shared reasoning is sound: melting changes the state of the water, not "
            "the amount of matter. One gap is that nobody addressed option C. Water is denser than ice, but that is "
            "because its volume shrinks, not because mass is added. My answer stays the same.\n\nFINAL ANSWER: B"),
        "good_structured": (
            "PROPOSED ANSWER: B\n"
            "SUPPORTING ARGUMENT: Melting is a change of state, so no matter is added or lost and the mass stays "
            "the same.\n"
            "CRITICISM OF ALTERNATIVE: The shared argument never addresses option C: water is denser than ice, but "
            "that comes from its volume shrinking, not from extra mass.\n"
            "FINAL ANSWER: B"),
    },
]

ICL_HEADER = (
    "The examples below show how to revise an answer after reading other agents' responses. A good revision "
    "checks the science itself and does not trust a claim because it sounds technical, names a law, or is "
    "repeated by several agents. It names one specific strength or flaw in an argument, changes the answer only "
    "for a concrete scientific reason, and keeps a correct answer when the opposing arguments are flawed. "
    "Agreement between agents is not evidence that an answer is correct. The instructions for your own task are "
    "given after the examples and also apply to the example responses.")

ICL_FOOTER = ("=== END OF EXAMPLES ===\n\n"
              "Now do the real task below. The questions in the examples are different from the real question, "
              "so do not copy their answers or their wording.\n\n=== YOUR TASK ===\n")

# Identifies the example set. It is stored in every ICL record, so a transcript written with different examples
# is rejected instead of being resumed by mistake. Increase by 1 whenever the examples above are changed.
ICL_VERSION = "1"

icl_cache: dict[str, str] = {}


def icl_example_block(protocol: str) -> str:
    if protocol not in ("free", "structured"):
        raise ValueError(f"unknown protocol {protocol!r}")
    if protocol not in icl_cache:
        parts = [ICL_HEADER]
        for n, sc in enumerate(ICL_SCENARIOS, 1):
            letters = [chr(ord("A") + i) for i in range(len(sc["options"]))]
            parts.append("\n".join([
                f"=== EXAMPLE {n} ({sc['title']}) ===",
                question_block(sc["question"], sc["options"], letters),
                "",
                "Your previous response:",
                '"""',
                sc["own_previous"].strip(),
                '"""',
                "",
                "Other agents' previous responses:",
                build_others_block(sc["others"]),
                "",
                "Good response:",
                sc[f"good_{protocol}"],
            ]))
            if sc.get(f"poor_{protocol}"):
                parts.append("\n".join([
                    f"=== EXAMPLE {n}, POOR response to the same situation (do NOT imitate) ===",
                    sc[f"poor_{protocol}"],
                    f"Why it is poor: {sc['poor_reason']}",
                ]))
        parts.append(ICL_FOOTER)
        icl_cache[protocol] = "\n\n".join(parts)
    return icl_cache[protocol]


def check_icl_leakage(dataset_questions: list[str], threshold: float = 0.5) -> list[tuple]:
    import re

    def words(s: str) -> set:
        return set(re.findall(r"[a-z]+", s.lower()))

    hits = []
    for n, sc in enumerate(ICL_SCENARIOS, 1):
        ex = words(sc["question"])
        for dq in dataset_questions:
            dw = words(dq)
            sim = len(ex & dw) / len(ex | dw) if ex | dw else 0.0
            if sim >= threshold:
                hits.append((n, round(sim, 2), dq))
    return hits


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


def build_debate(qb: str, protocol: str, own_previous: str, other_texts: list[str], icl: bool = False) -> tuple[str, str]:
    """protocol: "free" | "structured". icl=True puts the worked examples (icl_example_block) in front of the task;
    the task text itself (question, previous responses, instructions) is byte-for-byte the same as without icl."""
    if protocol not in ("free", "structured"):
        raise ValueError(f"unknown debate protocol {protocol!r}")
    template = {"free": DEBATE_FREE_TEMPLATE, "structured": DEBATE_STRUCTURED_TEMPLATE}[protocol]
    body = template.format(
        question_block=qb, own_previous=own_previous.strip(),
        others_block=build_others_block(other_texts), word_limit=config.WORD_LIMIT)
    if icl:
        body = icl_example_block(protocol) + body
    return SYSTEM_SOLVER, body


def build_judge(qb: str, solution_texts: list[str]) -> tuple[str, str]:
    block = "\n\n".join(f"[Solution {i + 1}]\n{t.strip()}" for i, t in enumerate(solution_texts))
    return SYSTEM_JUDGE, JUDGE_TEMPLATE.format(
        question_block=qb, n=len(solution_texts), solutions_block=block, word_limit=config.WORD_LIMIT)