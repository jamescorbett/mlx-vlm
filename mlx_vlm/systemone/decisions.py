"""Compile typed questions into reads, and reads back into typed answers.

Each question becomes one seeded canvas whose only free slot is the answer, so
every question in a request is answered in a single batched forward pass. The
primitives differ only in how the option labels are built and how the resulting
probability distribution is reported.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..structured_reads import TemplatePlan, resolve_template
from .schemas import OPTION_DIGITS, OPTION_LETTERS, Question


def render(value: Any) -> str:
    """Flatten structured state or instructions into prompt text."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True, default=str)


@dataclass
class CompiledQuestion:
    """A question paired with the read that answers it."""

    key: str
    kind: str
    plan: TemplatePlan
    labels: List[str]
    legend: Dict[str, str]


def _labels_for(kind: str, count: int) -> List[str]:
    """Pick the slot labels for a question type.

    Scores get digits because the scale has to be ordinal for the model to read
    it as one; everything else gets letters, which stay clearly distinct from
    any numbers in the state.
    """
    alphabet = OPTION_DIGITS if kind == "score" else OPTION_LETTERS
    if count > len(alphabet):
        raise ValueError(
            f"{count} {'levels' if kind == 'score' else 'options'} exceeds the "
            f"{len(alphabet)} single-token labels available for a canvas slot."
        )
    return list(alphabet[:count])


def _options(question: Question) -> tuple[List[str], Dict[str, str]]:
    """Return (display labels, legend) for a question's criteria."""
    if question.type == "noul":
        criteria = question.criteria or {}
        return (
            [str(criteria.get("true", "yes")), str(criteria.get("false", "no"))],
            {"true": str(criteria.get("true", "yes")),
             "false": str(criteria.get("false", "no"))},
        )
    if question.type == "choice":
        options = list(question.criteria or {})
        legend = {
            option: (question.criteria or {}).get(option) or option
            for option in options
        }
        return options, legend
    levels = list(question.criteria or [])
    return levels, {str(index): level for index, level in enumerate(levels)}


def compile_question(tokenizer, key: str, question: Question) -> CompiledQuestion:
    """Turn one typed question into a single-slot read."""
    labels, legend = _options(question)
    letters = _labels_for(question.type, len(labels))
    menu = " ".join(f"{letter}={label}" for letter, label in zip(letters, labels))
    template = f"Q: {render(question.instructions)} ({menu}) A: {{answer}}"
    plan = resolve_template(tokenizer, template, letters)
    return CompiledQuestion(
        key=key, kind=question.type, plan=plan, labels=labels, legend=legend
    )


def _confidence(probabilities: List[float]) -> float:
    """How far the distribution sits from uniform, in [0, 1].

    Reported separately from the winning probability because the two answer
    different questions: a 0.6/0.4 split over two options and a 0.6/0.2/0.2 over
    three share a top probability but not the same decisiveness.
    """
    count = len(probabilities)
    if count < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(count)))


def _is_bimodal(probabilities: List[float], floor: float = 0.15) -> bool:
    """True when mass sits on separated peaks.

    An expected value only summarises a distribution with one peak. If a scored
    read splits across both ends of the scale, the mean lands in a middle the
    model never chose, so the caller needs to know not to use it.
    """
    peaks = [
        index
        for index, value in enumerate(probabilities)
        if value >= floor
        and (index == 0 or value >= probabilities[index - 1])
        and (index == len(probabilities) - 1 or value >= probabilities[index + 1])
    ]
    return len(peaks) > 1


def build_answer(
    compiled: CompiledQuestion,
    probabilities: List[float],
    stderr: float,
) -> Dict[str, Any]:
    """Shape an averaged distribution into the answer for this question type."""
    confidence = _confidence(probabilities)

    if compiled.kind == "noul":
        return {
            "type": "noul",
            "noul": round(probabilities[0], 6),
            "confidence": round(confidence, 6),
            "stderr": round(stderr, 6),
        }

    if compiled.kind == "choice":
        by_option = {
            label: round(value, 6)
            for label, value in zip(compiled.labels, probabilities)
        }
        best = max(by_option, key=by_option.__getitem__)
        return {
            "type": "choice",
            "choice": best,
            "probabilities": by_option,
            "confidence": round(confidence, 6),
            "stderr": round(stderr, 6),
        }

    # A score is the distribution's expected level, so a confident answer
    # between two adjacent levels lands between them rather than snapping.
    by_index = {
        str(index): round(value, 6) for index, value in enumerate(probabilities)
    }
    score = sum(index * value for index, value in enumerate(probabilities))
    mode = max(range(len(probabilities)), key=probabilities.__getitem__)
    # The modal level is reported alongside the mean, and a split distribution
    # is flagged outright: either signal tells a caller the score is not a
    # summary they should act on.
    return {
        "type": "score",
        "score": round(score, 6),
        "mode": mode,
        "bimodal": _is_bimodal(probabilities),
        "legend": compiled.legend,
        "probabilities": by_index,
        "confidence": round(confidence, 6),
        "stderr": round(stderr, 6),
    }
