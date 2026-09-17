"""Structured reads with DiffusionGemma: bounded choices with confidences.

A *read* seeds the denoising canvas with an answer template whose answer slot is
left free, denoises it for a fixed number of steps, and reads the calibrated
log-probabilities at that slot. The model answers a multiple-choice question and
reports how sure it is, instead of generating free text you then have to parse.

    uv run examples/diffusion_structured_reads.py \
        --model google/diffusiongemma-26B-A4B-it \
        --question "What language is this snippet written in?" \
        --context "def f(x): return [i**2 for i in range(x)]" \
        --choices A B C \
        --labels Python Rust Haskell \
        --reads 5

Choices must be single tokens that differ in exactly one template position, so
single letters or digits are the reliable pick; ``--labels`` maps them back to
human-readable names in the output.
"""

from __future__ import annotations

import argparse

import mlx.core as mx

from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.structured_reads import ReadSession, resolve_template

# A template must not put punctuation immediately after the answer slot: the
# denoiser then competes to place that punctuation *in* the slot, which flattens
# the choice probabilities. "Answer: {answer}." measures visibly less sharply
# than the same template without the trailing period.
DEFAULT_TEMPLATE = "Answer: {answer}"


def build_prompt(processor, config, question, context, choices, labels):
    menu = "\n".join(f"{c}. {l}" for c, l in zip(choices, labels))
    body = f"{question}\n\n"
    if context:
        body += f"{context}\n\n"
    body += f"Choose one:\n{menu}"
    return apply_chat_template(processor, config, body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Diffusion model path or repo id")
    parser.add_argument(
        "--question",
        required=True,
        nargs="+",
        help="One or more questions; several share a single cached prompt",
    )
    parser.add_argument("--context", default="", help="Optional material to read")
    parser.add_argument(
        "--choices",
        nargs="+",
        default=["A", "B"],
        help="Single-token answer labels seeded into the canvas",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Human-readable name per choice (defaults to the choices)",
    )
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--reads",
        type=int,
        default=1,
        help="Reads to average; >1 gives an error bar over canvas noise",
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="Denoising steps per read"
    )
    parser.add_argument("--canvas-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    labels = args.labels or args.choices
    if len(labels) != len(args.choices):
        parser.error("--labels must have one entry per choice")

    model, processor = load(args.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    questions = args.question
    naming = dict(zip(args.choices, labels))

    if len(questions) == 1:
        prompt = build_prompt(
            processor, model.config, questions[0], args.context, args.choices, labels
        )
        templates = [args.template]
    else:
        # Several questions share one cached document: the document goes in the
        # prompt and each question rides in the seed canvas, so the prompt is
        # encoded once no matter how many questions follow.
        menu = " ".join(f"{c}={l}" for c, l in zip(args.choices, labels))
        prompt = build_prompt(
            processor, model.config,
            "Read this and answer the questions.", args.context, args.choices, labels,
        )
        templates = [f"Q: {q} ({menu}) A: {{answer}}" for q in questions]

    session = ReadSession(
        model, processor, tokenizer, mx.array([tokenizer.encode(prompt)])
    )

    for question, template in zip(questions, templates):
        plan = resolve_template(tokenizer, template, args.choices)
        decision = session.decide(
            plan,
            reads=args.reads,
            canvas_length=args.canvas_length,
            steps=args.steps,
            temperature=args.temperature,
            seed=args.seed,
        )
        print(f"\nQ: {question}")
        print(f"A: {naming[decision.choice]}  ({decision.choice})")
        print(f"   confidence {decision.probability:.3f} ± {decision.stderr:.3f}")
        print(f"   margin over runner-up {decision.margin:.3f}\n")
        for choice, probability in sorted(
            decision.probabilities.items(), key=lambda kv: -kv[1]
        ):
            bar = "█" * round(probability * 40)
            print(f"   {choice} {naming[choice]:<20} {probability:6.3f} {bar}")


if __name__ == "__main__":
    main()
