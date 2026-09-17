"""Structured reads for masked-diffusion language models.

A *read* treats the denoising canvas as a classifier instead of a text
generator. The caller supplies an answer template with a single free slot, the
engine denoises that slot for a fixed number of steps, and the temperature-1.0
log-probabilities at the slot are read off as calibrated confidences over the
allowed choices.

This is the mlx-vlm counterpart to vLLM's ``diffusion_reads`` example. vLLM
exposes the primitives as per-request ``vllm_xargs`` because it serves many
requests in batched slots; mlx-vlm's diffusion engine runs one request at a
time, so the same capability is a local Python API built on three engine
parameters:

``diffusion_seed_canvas``
    Pins template tokens and leaves :data:`~mlx_vlm.generate.diffusion.
    DIFFUSION_FREE_SLOT` positions for the denoiser.
``diffusion_read_only``
    Emits the argmax canvas after the step cap and ends the request.
``logprob_token_ids``
    Returns per-position log-probabilities for the choice tokens.

Typical use::

    plan = resolve_template(tokenizer, "Answer: {answer}", ["A", "B"])
    decision = decide(model, processor, tokenizer, prompt_ids, plan, reads=5)
    print(decision.choice, decision.probability, decision.stderr)

Template choice matters. Avoid punctuation directly after the answer slot: the
denoiser competes to place that punctuation in the slot itself, which flattens
the reported confidences without changing which choice wins.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx

from .generate.diffusion import (
    DIFFUSION_FREE_SLOT,
    _apply_seed_canvas,
    _normalize_seed_canvas,
    stream_diffusion_generate,
)

DEFAULT_READ_STEPS = 1
DEFAULT_READS = 1
ANSWER_PLACEHOLDER = "{answer}"


@dataclass
class TemplatePlan:
    """A resolved answer template ready to be seeded onto a canvas.

    ``base_ids`` is the tokenized template with the answer slot still occupied
    by an arbitrary choice; ``slot_index`` is the one position that varies
    between choices, and ``choice_token_ids[i]`` is the token that position
    takes for ``choices[i]``.
    """

    template: str
    choices: List[str]
    base_ids: List[int]
    slot_index: int
    choice_token_ids: List[int]

    @property
    def width(self) -> int:
        return len(self.base_ids)

    def seed_canvas(self, canvas_length: Optional[int] = None) -> List[int]:
        """Template token ids with the answer slot left free."""
        canvas_length = canvas_length or self.width
        if canvas_length < self.width:
            raise ValueError(
                f"canvas_length {canvas_length} is narrower than the "
                f"{self.width}-token template."
            )
        seed = list(self.base_ids) + [DIFFUSION_FREE_SLOT] * (
            canvas_length - self.width
        )
        seed[self.slot_index] = DIFFUSION_FREE_SLOT
        return seed


@dataclass
class ReadResult:
    """One denoising read: the probability mass on each choice."""

    probabilities: List[float]
    logprobs: List[float]
    choices: List[str]
    canvas_token_ids: List[int] = field(default_factory=list)

    @property
    def choice(self) -> str:
        return self.choices[max(range(len(self.probabilities)), key=self.probabilities.__getitem__)]


@dataclass
class Decision:
    """An aggregate decision over one or more reads."""

    choice: str
    probability: float
    stderr: float
    probabilities: Dict[str, float]
    reads: List[ReadResult] = field(default_factory=list)

    @property
    def margin(self) -> float:
        """Gap between the winner and the runner-up."""
        ordered = sorted(self.probabilities.values(), reverse=True)
        return ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]


def _encode(tokenizer, text: str) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in ids]


def resolve_template(
    tokenizer,
    template: str,
    choices: Sequence[str],
    *,
    placeholder: str = ANSWER_PLACEHOLDER,
) -> TemplatePlan:
    """Tokenize ``template`` for each choice and locate the single answer slot.

    A template only works as a read if swapping the choice changes exactly one
    token: the denoiser fills one slot, so any choice needing two or more tokens
    (or shifting the tokens around it) cannot be distinguished from the canvas.
    Rejecting those here is what makes a read's probabilities meaningful.
    """
    choices = [str(choice) for choice in choices]
    if len(choices) < 2:
        raise ValueError("A read needs at least two choices.")
    if len(set(choices)) != len(choices):
        raise ValueError(f"Choices must be unique, got {choices!r}.")
    if placeholder not in template:
        raise ValueError(
            f"Template {template!r} does not contain the placeholder {placeholder!r}."
        )

    variants = [_encode(tokenizer, template.replace(placeholder, c)) for c in choices]

    widths = {len(variant) for variant in variants}
    if len(widths) != 1:
        raise ValueError(
            "Each choice must tokenize to the same template width; got widths "
            f"{sorted(widths)} for choices {choices!r}. Try padding the choices "
            "to a common shape (e.g. single letters or digits)."
        )

    base = variants[0]
    differing = [
        index
        for index in range(len(base))
        if len({variant[index] for variant in variants}) > 1
    ]
    if len(differing) != 1:
        raise ValueError(
            f"Choices {choices!r} differ at {len(differing)} token positions "
            f"({differing}); a read requires exactly one. Pick choices that are "
            "single tokens in this tokenizer."
        )

    slot_index = differing[0]
    choice_token_ids = [variant[slot_index] for variant in variants]
    if len(set(choice_token_ids)) != len(choice_token_ids):
        raise ValueError(
            f"Choices {choices!r} collapse to duplicate tokens at the answer slot."
        )

    return TemplatePlan(
        template=template,
        choices=choices,
        base_ids=base,
        slot_index=slot_index,
        choice_token_ids=choice_token_ids,
    )


def _softmax(logprobs: Sequence[float]) -> List[float]:
    """Renormalize choice log-probabilities over the allowed choices only."""
    highest = max(logprobs)
    weights = [math.exp(value - highest) for value in logprobs]
    total = sum(weights)
    return [weight / total for weight in weights]


class ReadSession:
    """A prompt encoded once, then read many times.

    Encoding the prompt dominates the cost of a read: the denoising itself is a
    single pass over a canvas a few tokens wide, while the prefill covers the
    whole prompt. Running N reads through :func:`read_once` therefore pays for N
    identical prefills. A session pays once and reuses the KV cache, which is
    safe because a read never writes to it — only multi-block generation
    appends, and a read ends before reaching that path.

    A session caches one *prompt*, so varying the question means putting the
    question in the seed canvas rather than the prompt. Put the document in the
    prompt and the question in the template, and one prefill serves every
    question asked about it::

        session = ReadSession(model, processor, tokenizer, document_ids)
        for question in questions:
            plan = resolve_template(
                tokenizer, f"Q: {question} (A=yes B=no) A: {{answer}}", ["A", "B"]
            )
            print(session.decide(plan, reads=5))

    Measured on ``diffusiongemma-26B-A4B-it``: 18 reads over 6 questions against
    one 89-token document cost a single prefill, and a 1833-token prompt read 10
    times went from 22.9s to 2.1s.
    """

    def __init__(
        self,
        model,
        processor,
        tokenizer,
        input_ids: mx.array,
        *,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        mm_token_type_ids: Optional[mx.array] = None,
        prefill_step_size: Optional[int] = None,
    ):
        if input_ids.shape[0] != 1:
            raise ValueError("Structured reads only support batch size 1.")

        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.pixel_values = pixel_values
        self.attention_mask = attention_mask
        self.prompt_tokens = int(input_ids.shape[1])

        cache = model.make_cache()
        self.cache = model.diffusion_prefill_cache(
            input_ids,
            attention_mask=attention_mask,
            cache=cache,
            pixel_values=pixel_values,
            mm_token_type_ids=mm_token_type_ids,
            prefill_step_size=prefill_step_size,
            chunk_prefill=False,
        )
        mx.eval([c.state for c in self.cache])

    def read(
        self,
        plan: TemplatePlan,
        *,
        canvas_length: Optional[int] = None,
        steps: int = DEFAULT_READ_STEPS,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        **engine_kwargs: Any,
    ) -> ReadResult:
        """Run one read against the cached prompt."""
        return _read_with_cache(
            self.model,
            self.processor,
            self.tokenizer,
            self.input_ids,
            plan,
            prompt_cache=self.cache,
            pixel_values=self.pixel_values,
            attention_mask=self.attention_mask,
            canvas_length=canvas_length,
            steps=steps,
            temperature=temperature,
            seed=seed,
            **engine_kwargs,
        )

    def read_batch(
        self,
        plans: Sequence[TemplatePlan],
        *,
        canvas_length: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[ReadResult]:
        """Denoise several reads in one forward pass against the cached prompt.

        Every canvas attends to the same cached prefix, so the batch costs one
        pass instead of ``len(plans)``. Canvases are padded to a common width,
        which means a batch runs at the width of its widest template — group
        similarly sized questions together to avoid paying for the outlier.

        Only single-step reads are batched. Multi-step denoising runs the
        sampler's accept/resample loop, which this path does not reproduce;
        :meth:`read` handles those so the two can never quietly disagree.
        """
        plans = list(plans)
        if not plans:
            return []
        if seed is not None:
            mx.random.seed(seed)

        width = canvas_length or max(plan.width for plan in plans)
        vocab_size = int(self.model.config.text_config.vocab_size)
        dtype = self.input_ids.dtype

        canvases = []
        for plan in plans:
            values, mask = _normalize_seed_canvas(
                plan.seed_canvas(width), width, vocab_size, dtype
            )
            noise = mx.random.randint(0, vocab_size, (1, width)).astype(dtype)
            canvases.append(_apply_seed_canvas(noise, values, mask))
        batch = mx.concatenate(canvases, axis=0)

        widened = _broadcast_cache(self.cache, len(plans))
        masks = self.model.diffusion_decoder_masks(batch, widened, None)
        logits = self.model.diffusion_decoder_logits(
            batch,
            cache=widened,
            self_conditioning=None,
            decoder_attention_mask=masks,
        )
        logits = logits.astype(mx.float32)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        argmax_canvas = mx.argmax(logits, axis=-1)
        mx.eval(log_probs, argmax_canvas)

        results = []
        for index, plan in enumerate(plans):
            selection = mx.array(plan.choice_token_ids, dtype=mx.int32)
            slot_logprobs = [
                float(value) for value in log_probs[index, plan.slot_index][selection]
            ]
            emitted = [int(token) for token in argmax_canvas[index].tolist()]
            # Pinned template slots are fixed by construction, so report them as
            # seeded rather than as whatever the denoiser scored highest there.
            for position, seeded in enumerate(plan.seed_canvas(width)):
                if seeded != DIFFUSION_FREE_SLOT:
                    emitted[position] = seeded
            results.append(
                ReadResult(
                    probabilities=_softmax(slot_logprobs),
                    logprobs=slot_logprobs,
                    choices=list(plan.choices),
                    canvas_token_ids=emitted,
                )
            )
        return results

    def decide(
        self,
        plan: TemplatePlan,
        *,
        reads: int = DEFAULT_READS,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Decision:
        """Average several reads against the cached prompt."""
        return _aggregate(
            plan,
            reads,
            seed,
            lambda read_seed: self.read(plan, seed=read_seed, **kwargs),
        )

    def decide_batch(
        self,
        plan: TemplatePlan,
        *,
        reads: int = DEFAULT_READS,
        canvas_length: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Decision:
        """Average several reads taken in a single batched pass.

        The repeats of one question are the natural batch: they share a template
        and differ only in canvas noise, so they pad to the same width with
        nothing wasted. Single-step only, like :meth:`read_batch`.
        """
        if reads < 1:
            raise ValueError("reads must be a positive integer.")
        results = self.read_batch(
            [plan] * reads, canvas_length=canvas_length, seed=seed
        )
        return _summarize(plan, results)


def _broadcast_cache(cache, batch_size: int):
    """Widen a batch-1 prompt cache so one cached prefix serves N canvases.

    The entries are copied before their state is replaced, so the caller's cache
    is left alone. Broadcasting rather than tiling keeps this free: every canvas
    attends to the same prefix, and a read never writes back.
    """
    widened = []
    for entry in cache:
        keys, values = entry.state
        if keys is None:
            widened.append(entry)
            continue
        if keys.shape[0] != 1:
            raise ValueError(
                f"Expected a batch-1 prompt cache, got batch {keys.shape[0]}."
            )
        clone = copy.copy(entry)
        clone.state = (
            mx.broadcast_to(keys, (batch_size,) + keys.shape[1:]),
            mx.broadcast_to(values, (batch_size,) + values.shape[1:]),
        )
        widened.append(clone)
    return widened


def _read_with_cache(
    model,
    processor,
    tokenizer,
    input_ids: mx.array,
    plan: TemplatePlan,
    *,
    prompt_cache=None,
    canvas_length: Optional[int] = None,
    steps: int = DEFAULT_READ_STEPS,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    **engine_kwargs: Any,
) -> ReadResult:
    if steps < 1:
        raise ValueError("steps must be a positive integer.")
    if seed is not None:
        mx.random.seed(seed)

    seed_canvas = plan.seed_canvas(canvas_length)

    terminal = None
    stream = stream_diffusion_generate(
        model,
        processor,
        tokenizer,
        input_ids,
        engine_kwargs.pop("pixel_values", None),
        engine_kwargs.pop("attention_mask", None),
        max_tokens=len(seed_canvas),
        skip_special_token_ids=engine_kwargs.pop("skip_special_token_ids", []),
        temperature=temperature,
        max_denoising_steps=steps,
        diffusion_seed_canvas=seed_canvas,
        diffusion_read_only=True,
        logprob_token_ids=plan.choice_token_ids,
        prompt_cache=prompt_cache,
        **engine_kwargs,
    )
    try:
        for result in stream:
            if result.diffusion_canvas_logprobs is not None:
                terminal = result
    finally:
        stream.close()

    if terminal is None or terminal.diffusion_canvas_logprobs is None:
        raise RuntimeError("The diffusion engine returned no canvas log-probabilities.")

    slot_logprobs = [
        float(value) for value in terminal.diffusion_canvas_logprobs[plan.slot_index]
    ]
    return ReadResult(
        probabilities=_softmax(slot_logprobs),
        logprobs=slot_logprobs,
        choices=list(plan.choices),
        canvas_token_ids=list(terminal.token_ids or []),
    )


def read_once(
    model,
    processor,
    tokenizer,
    input_ids: mx.array,
    plan: TemplatePlan,
    **kwargs: Any,
) -> ReadResult:
    """Run a single read, encoding the prompt for this call only.

    Use :class:`ReadSession` when reading the same prompt more than once.
    """
    return _read_with_cache(model, processor, tokenizer, input_ids, plan, **kwargs)


def _aggregate(plan: TemplatePlan, reads: int, seed: Optional[int], run) -> Decision:
    if reads < 1:
        raise ValueError("reads must be a positive integer.")

    # Each read needs its own seed: identical seeds would give identical canvas
    # noise, collapsing the spread the error bar is meant to measure.
    results = [
        run(None if seed is None else seed + index) for index in range(reads)
    ]
    return _summarize(plan, results)


def _summarize(plan: TemplatePlan, results: List[ReadResult]) -> Decision:
    averaged = [
        sum(result.probabilities[i] for result in results) / len(results)
        for i in range(len(plan.choices))
    ]
    best = max(range(len(averaged)), key=averaged.__getitem__)

    if len(results) > 1:
        samples = [result.probabilities[best] for result in results]
        mean = sum(samples) / len(samples)
        variance = sum((value - mean) ** 2 for value in samples) / (len(samples) - 1)
        stderr = math.sqrt(variance / len(samples))
    else:
        stderr = 0.0

    return Decision(
        choice=plan.choices[best],
        probability=averaged[best],
        stderr=stderr,
        probabilities=dict(zip(plan.choices, averaged)),
        reads=results,
    )


def decide(
    model,
    processor,
    tokenizer,
    input_ids: mx.array,
    plan: TemplatePlan,
    *,
    reads: int = DEFAULT_READS,
    canvas_length: Optional[int] = None,
    steps: int = DEFAULT_READ_STEPS,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    pixel_values: Optional[mx.array] = None,
    attention_mask: Optional[mx.array] = None,
    **engine_kwargs: Any,
) -> Decision:
    """Average several reads into one decision with an error bar.

    Repeated reads differ because the unseeded canvas slots start from fresh
    noise, so the spread across reads is a genuine uncertainty estimate rather
    than sampling jitter.

    The prompt is encoded once and reused across the reads. Hold the
    :class:`ReadSession` yourself to extend that reuse across several questions.
    """
    session = ReadSession(
        model,
        processor,
        tokenizer,
        input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
    )
    return session.decide(
        plan,
        reads=reads,
        seed=seed,
        canvas_length=canvas_length,
        steps=steps,
        temperature=temperature,
        **engine_kwargs,
    )
