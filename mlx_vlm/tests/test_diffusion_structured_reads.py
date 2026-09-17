"""Structured reads on masked-diffusion models.

Exercises the engine primitives (seed canvas, read-only emit, calibrated
log-probabilities) against a stub diffusion model, plus the template resolution
that turns a question into a one-slot read.
"""

import math
import unittest
from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.generate.diffusion import (
    DIFFUSION_FREE_SLOT,
    _apply_seed_canvas,
    _diffusion_canvas_logprobs,
    _normalize_seed_canvas,
    stream_diffusion_generate,
)
from mlx_vlm.structured_reads import (
    ReadSession,
    TemplatePlan,
    decide,
    read_once,
    resolve_template,
)

VOCAB = 256
# Wide enough to hold the test templates with free slots to spare.
CANVAS = 24


class StubDetokenizer:
    def __init__(self):
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token, skip_special_token_ids=None):
        self.last_segment = chr(int(token))

    def finalize(self):
        self.last_segment = ""


class StubTokenizer:
    """One character per token, id == ord(char)."""

    def __init__(self):
        self.stopping_criteria = lambda token_id: False
        self.detokenizer = StubDetokenizer()

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(int(t)) for t in token_ids)


class StubCache:
    state = None


class StubDiffusionModel:
    """Minimal model exposing the engine's diffusion contract.

    The decoder returns a fixed logit field so every assertion about seeding and
    log-probabilities is deterministic: ``favoured_token`` gets the highest
    logit everywhere, which is what the free slots denoise to.
    """

    def __init__(self, favoured_token, runner_up_token):
        self.favoured_token = favoured_token
        self.runner_up_token = runner_up_token
        self.canvases_seen = []
        self.prefill_calls = 0
        self.cache_writes = 0
        self.config = SimpleNamespace(
            canvas_length=CANVAS,
            mask_token_id=0,
            generation_config={},
            text_config=SimpleNamespace(vocab_size=VOCAB),
        )

    def make_cache(self, max_size=None):
        return [StubCache()]

    def diffusion_prepare_self_conditioning(self):
        return None

    def diffusion_self_conditioning(self, logits, context):
        return None

    def diffusion_prefill_cache(self, input_ids, **kwargs):
        self.prefill_calls += 1
        return kwargs.get("cache") or [StubCache()]

    def diffusion_update_cache(self, input_ids, *, cache):
        self.cache_writes += 1
        return cache

    def diffusion_decoder_masks(self, canvas, cache, mask):
        return None

    def diffusion_decoder_logits(
        self, canvas_ids, cache=None, self_conditioning=None, decoder_attention_mask=None
    ):
        self.canvases_seen.append([int(t) for t in canvas_ids[0].tolist()])
        length = canvas_ids.shape[1]
        logits = mx.zeros((1, length, VOCAB))
        logits[:, :, self.favoured_token] = 4.0
        logits[:, :, self.runner_up_token] = 2.0
        return logits


def run_read(model, plan, **kwargs):
    tokenizer = StubTokenizer()
    return read_once(
        model,
        tokenizer,
        tokenizer,
        mx.array([[1, 2, 3]]),
        plan,
        **kwargs,
    )


class TestSeedCanvasHelpers(unittest.TestCase):
    def test_pins_only_the_seeded_slots(self):
        values, mask = _normalize_seed_canvas([10, 11, DIFFUSION_FREE_SLOT, 13], 6, VOCAB, mx.int32)
        canvas = mx.full((1, 6), 99, dtype=mx.int32)
        self.assertEqual(
            _apply_seed_canvas(canvas, values, mask).tolist()[0],
            [10, 11, 99, 13, 99, 99],
        )

    def test_none_seed_is_a_passthrough(self):
        values, mask = _normalize_seed_canvas(None, 6, VOCAB, mx.int32)
        canvas = mx.full((1, 6), 7, dtype=mx.int32)
        self.assertIsNone(values)
        self.assertEqual(_apply_seed_canvas(canvas, values, mask).tolist(), canvas.tolist())

    def test_trailing_slots_are_free(self):
        _, mask = _normalize_seed_canvas([5, 6], 5, VOCAB, mx.int32)
        self.assertEqual(mask.tolist()[0], [True, True, False, False, False])

    def test_rejects_out_of_vocabulary_and_oversized_seeds(self):
        with self.assertRaises(ValueError):
            _normalize_seed_canvas([VOCAB + 1], 4, VOCAB, mx.int32)
        with self.assertRaises(ValueError):
            _normalize_seed_canvas([1, 2, 3], 2, VOCAB, mx.int32)

    def test_logprobs_are_a_plain_log_softmax(self):
        logits = mx.random.normal((1, 4, VOCAB))
        got = _diffusion_canvas_logprobs(logits, [3, 9])
        self.assertEqual(got.shape, (4, 2))
        want = logits[0, 2] - mx.logsumexp(logits[0, 2])
        self.assertAlmostEqual(float(got[2, 1]), float(want[9]), places=4)


class TestTemplateResolution(unittest.TestCase):
    def setUp(self):
        self.tokenizer = StubTokenizer()

    def test_locates_the_single_answer_slot(self):
        plan = resolve_template(self.tokenizer, "Answer: {answer}.", ["A", "B"])
        self.assertEqual(plan.slot_index, len("Answer: "))
        self.assertEqual(plan.choice_token_ids, [ord("A"), ord("B")])

    def test_seed_canvas_frees_the_answer_slot(self):
        plan = resolve_template(self.tokenizer, "Answer: {answer}.", ["A", "B"])
        seed = plan.seed_canvas()
        self.assertEqual(seed[plan.slot_index], DIFFUSION_FREE_SLOT)
        self.assertEqual(seed[0], ord("A"))
        self.assertEqual(len(seed), plan.width)

    def test_seed_canvas_pads_free_slots_to_the_requested_width(self):
        plan = resolve_template(self.tokenizer, "Answer: {answer}.", ["A", "B"])
        seed = plan.seed_canvas(plan.width + 3)
        self.assertEqual(seed[-3:], [DIFFUSION_FREE_SLOT] * 3)
        with self.assertRaises(ValueError):
            plan.seed_canvas(plan.width - 1)

    def test_rejects_multi_token_choices(self):
        # "yes"/"no" tokenize to different widths, so no single slot distinguishes them.
        with self.assertRaises(ValueError):
            resolve_template(self.tokenizer, "Answer: {answer}.", ["yes", "no"])

    def test_rejects_degenerate_choice_sets(self):
        with self.assertRaises(ValueError):
            resolve_template(self.tokenizer, "Answer: {answer}.", ["A"])
        with self.assertRaises(ValueError):
            resolve_template(self.tokenizer, "Answer: {answer}.", ["A", "A"])
        with self.assertRaises(ValueError):
            resolve_template(self.tokenizer, "no placeholder", ["A", "B"])


class TestReadOnlyEngine(unittest.TestCase):
    def setUp(self):
        self.tokenizer = StubTokenizer()
        self.plan = resolve_template(self.tokenizer, "Answer: {answer}.", ["A", "B"])
        self.model = StubDiffusionModel(
            favoured_token=self.plan.choice_token_ids[0],
            runner_up_token=self.plan.choice_token_ids[1],
        )

    def _stream(self, **overrides):
        kwargs = dict(
            max_tokens=self.plan.width,
            skip_special_token_ids=[],
            temperature=0.0,
            max_denoising_steps=1,
            diffusion_seed_canvas=self.plan.seed_canvas(),
            diffusion_read_only=True,
            logprob_token_ids=self.plan.choice_token_ids,
        )
        kwargs.update(overrides)
        return list(
            stream_diffusion_generate(
                self.model,
                self.tokenizer,
                self.tokenizer,
                mx.array([[1, 2, 3]]),
                None,
                None,
                **kwargs,
            )
        )

    def test_read_only_emits_exactly_one_canvas_and_stops(self):
        results = self._stream()
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result.diffusion_block_complete)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(len(result.token_ids), self.plan.width)

    def test_read_only_runs_a_single_denoising_pass(self):
        self._stream()
        self.assertEqual(len(self.model.canvases_seen), 1)

    def test_template_tokens_are_pinned_and_the_slot_is_denoised(self):
        result = self._stream()[0]
        emitted = result.token_ids
        for index, seeded in enumerate(self.plan.seed_canvas()):
            if seeded != DIFFUSION_FREE_SLOT:
                self.assertEqual(emitted[index], seeded, f"slot {index} drifted")
        # The free slot took the model's favoured token, not the template's.
        self.assertEqual(emitted[self.plan.slot_index], self.model.favoured_token)

    def test_the_decoder_sees_the_seed_on_its_first_pass(self):
        self._stream()
        seen = self.model.canvases_seen[0]
        for index, seeded in enumerate(self.plan.seed_canvas()):
            if seeded != DIFFUSION_FREE_SLOT:
                self.assertEqual(seen[index], seeded)

    def test_logprobs_cover_every_position_and_choice(self):
        result = self._stream()[0]
        self.assertEqual(len(result.diffusion_canvas_logprobs), self.plan.width)
        self.assertEqual(len(result.diffusion_canvas_logprobs[0]), 2)
        self.assertEqual(result.diffusion_logprob_token_ids, self.plan.choice_token_ids)

    def test_logprobs_ignore_the_denoising_temperature_schedule(self):
        # Logits are 4.0 and 2.0, so a calibrated read must show a gap of
        # exactly 2.0 regardless of the schedule the sampler used.
        result = self._stream()[0]
        first, second = result.diffusion_canvas_logprobs[self.plan.slot_index]
        self.assertAlmostEqual(first - second, 2.0, places=4)

    def test_non_read_only_generation_is_unaffected(self):
        results = self._stream(diffusion_read_only=False, logprob_token_ids=None)
        self.assertGreater(len(results), 1)
        self.assertIsNone(results[-1].diffusion_canvas_logprobs)

    def test_rejects_invalid_read_parameters(self):
        with self.assertRaises(ValueError):
            self._stream(logprob_token_ids=[VOCAB + 5])
        with self.assertRaises(ValueError):
            self._stream(logprob_token_ids=[])
        with self.assertRaises(ValueError):
            self._stream(diffusion_seed_canvas=[1] * (CANVAS + 1))


class TestReadApi(unittest.TestCase):
    def setUp(self):
        self.tokenizer = StubTokenizer()
        self.plan = resolve_template(self.tokenizer, "Answer: {answer}.", ["A", "B"])
        self.model = StubDiffusionModel(
            favoured_token=self.plan.choice_token_ids[0],
            runner_up_token=self.plan.choice_token_ids[1],
        )

    def test_read_once_reports_calibrated_probabilities(self):
        read = run_read(self.model, self.plan)
        self.assertEqual(read.choice, "A")
        # softmax over logits 4.0 and 2.0 renormalized across the two choices.
        expected = 1.0 / (1.0 + math.exp(-2.0))
        self.assertAlmostEqual(read.probabilities[0], expected, places=4)
        self.assertAlmostEqual(sum(read.probabilities), 1.0, places=6)

    def test_read_once_accepts_a_wider_canvas(self):
        read = run_read(self.model, self.plan, canvas_length=CANVAS)
        self.assertEqual(len(read.canvas_token_ids), CANVAS)
        self.assertEqual(read.choice, "A")

    def test_decide_aggregates_reads_with_an_error_bar(self):
        decision = decide(
            self.model,
            self.tokenizer,
            self.tokenizer,
            mx.array([[1, 2, 3]]),
            self.plan,
            reads=3,
        )
        self.assertEqual(decision.choice, "A")
        self.assertEqual(len(decision.reads), 3)
        self.assertAlmostEqual(sum(decision.probabilities.values()), 1.0, places=6)
        # The stub is deterministic, so repeated reads agree exactly.
        self.assertAlmostEqual(decision.stderr, 0.0, places=9)
        self.assertGreater(decision.margin, 0.0)

    def test_decide_encodes_the_prompt_only_once(self):
        decide(
            self.model,
            self.tokenizer,
            self.tokenizer,
            mx.array([[1, 2, 3]]),
            self.plan,
            reads=4,
        )
        self.assertEqual(self.model.prefill_calls, 1)
        self.assertEqual(len(self.model.canvases_seen), 4)

    def test_session_reuses_one_prefill_across_questions(self):
        session = ReadSession(
            self.model, self.tokenizer, self.tokenizer, mx.array([[1, 2, 3]])
        )
        self.assertEqual(self.model.prefill_calls, 1)
        other = resolve_template(self.tokenizer, "Verdict: {answer}", ["A", "B"])
        session.decide(self.plan, reads=2)
        session.decide(other, reads=2)
        self.assertEqual(self.model.prefill_calls, 1)
        self.assertEqual(len(self.model.canvases_seen), 4)

    def test_session_reads_match_a_standalone_read(self):
        session = ReadSession(
            self.model, self.tokenizer, self.tokenizer, mx.array([[1, 2, 3]])
        )
        standalone = run_read(self.model, self.plan)
        pooled = session.read(self.plan)
        self.assertEqual(pooled.choice, standalone.choice)
        for got, want in zip(pooled.logprobs, standalone.logprobs):
            self.assertAlmostEqual(got, want, places=5)

    def test_session_does_not_mutate_the_cached_prompt(self):
        session = ReadSession(
            self.model, self.tokenizer, self.tokenizer, mx.array([[1, 2, 3]])
        )
        before = self.model.cache_writes
        session.decide(self.plan, reads=3)
        self.assertEqual(self.model.cache_writes, before)

    def test_prompt_cache_is_refused_for_ordinary_generation(self):
        # Generation appends each block to the cache, which would corrupt a
        # prefix the caller still intends to reuse.
        with self.assertRaises(ValueError):
            list(
                stream_diffusion_generate(
                    self.model,
                    self.tokenizer,
                    self.tokenizer,
                    mx.array([[1, 2, 3]]),
                    None,
                    None,
                    max_tokens=4,
                    skip_special_token_ids=[],
                    diffusion_read_only=False,
                    prompt_cache=self.model.make_cache(),
                )
            )

    def test_decide_rejects_a_non_positive_read_count(self):
        with self.assertRaises(ValueError):
            decide(
                self.model,
                self.tokenizer,
                self.tokenizer,
                mx.array([[1, 2, 3]]),
                self.plan,
                reads=0,
            )


if __name__ == "__main__":
    unittest.main()
