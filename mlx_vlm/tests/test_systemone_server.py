"""The System One endpoint: typed questions in, typed decisions out.

Runs against a stub diffusion model, so the assertions are about the interface
contract — batching, state caching, answer shapes — rather than model quality.
"""

import unittest
from types import SimpleNamespace

import mlx.core as mx
from fastapi.testclient import TestClient

from mlx_vlm.systemone.app import StateCache, create_app
from mlx_vlm.systemone.decisions import _confidence, compile_question
from mlx_vlm.systemone.schemas import Question

VOCAB = 512
CANVAS = 256


class StubDetokenizer:
    def __init__(self):
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token, skip_special_token_ids=None):
        self.last_segment = ""

    def finalize(self):
        self.last_segment = ""


class StubTokenizer:
    def __init__(self):
        self.stopping_criteria = lambda t: False
        self.detokenizer = StubDetokenizer()

    def encode(self, text, add_special_tokens=False):
        return [min(ord(c), VOCAB - 1) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(t)) for t in ids)


class StubCache:
    def __init__(self):
        self.state = (mx.zeros((1, 2, 3, 4)), mx.zeros((1, 2, 3, 4)))


class StubModel:
    """Favours the first option letter so answers are deterministic."""

    def __init__(self):
        self.forward_passes = 0
        self.prefills = 0
        self.config = SimpleNamespace(
            canvas_length=CANVAS,
            mask_token_id=0,
            generation_config={},
            text_config=SimpleNamespace(vocab_size=VOCAB),
            model_type="diffusion_gemma",
        )

    def make_cache(self, max_size=None):
        return [StubCache()]

    def diffusion_prefill_cache(self, input_ids, **kwargs):
        self.prefills += 1
        return kwargs.get("cache") or [StubCache()]

    def diffusion_decoder_masks(self, canvas, cache, mask):
        return None

    def diffusion_decoder_logits(self, canvas_ids, **kwargs):
        self.forward_passes += 1
        logits = mx.zeros((canvas_ids.shape[0], canvas_ids.shape[1], VOCAB))
        logits[:, :, ord("A")] = 5.0
        logits[:, :, ord("B")] = 3.0
        logits[:, :, ord("C")] = 1.0
        return logits


def build_client():
    model = StubModel()
    tokenizer = StubTokenizer()
    app = create_app(model, tokenizer, "stub-diffusion")
    return TestClient(app), model


def post(client, **body):
    body.setdefault("state", "some state")
    return client.post("/v1/systemone", json=body)


class TestSystemOneEndpoint(unittest.TestCase):
    def setUp(self):
        self.client, self.model = build_client()

    def test_health_reports_the_model_and_cache(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["model"], "stub-diffusion")
        self.assertIn("state_cache", body)

    def test_noul_returns_a_probability(self):
        body = post(
            self.client,
            questions={"urgent": {"type": "noul", "instructions": "Is it urgent?"}},
        ).json()
        answer = body["answers"]["urgent"]
        self.assertEqual(answer["type"], "noul")
        self.assertGreaterEqual(answer["noul"], 0.0)
        self.assertLessEqual(answer["noul"], 1.0)

    def test_choice_returns_a_full_distribution(self):
        body = post(
            self.client,
            questions={
                "team": {
                    "type": "choice",
                    "instructions": "Who handles this?",
                    "criteria": {"billing": "money", "support": "help", "sales": None},
                }
            },
        ).json()
        answer = body["answers"]["team"]
        self.assertEqual(answer["type"], "choice")
        self.assertEqual(set(answer["probabilities"]), {"billing", "support", "sales"})
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0, places=4)
        self.assertIn(answer["choice"], answer["probabilities"])

    def test_score_returns_expected_value_legend_and_mode(self):
        body = post(
            self.client,
            questions={
                "sev": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["low", "medium", "high"],
                }
            },
        ).json()
        answer = body["answers"]["sev"]
        self.assertEqual(answer["type"], "score")
        self.assertEqual(answer["legend"], {"0": "low", "1": "medium", "2": "high"})
        self.assertGreaterEqual(answer["score"], 0.0)
        self.assertLessEqual(answer["score"], 2.0)
        self.assertIn(answer["mode"], (0, 1, 2))

    def test_every_question_is_answered_in_one_forward_pass(self):
        before = self.model.forward_passes
        body = post(
            self.client,
            reads=4,
            questions={
                "a": {"type": "noul", "instructions": "One?"},
                "b": {"type": "noul", "instructions": "Two?"},
                "c": {
                    "type": "choice",
                    "instructions": "Three?",
                    "criteria": {"x": None, "y": None},
                },
            },
        ).json()
        self.assertEqual(len(body["answers"]), 3)
        self.assertEqual(self.model.forward_passes - before, 1)
        self.assertEqual(body["usage"]["forward_passes"], 1)

    def test_a_repeated_state_skips_the_prefill(self):
        questions = {"q": {"type": "noul", "instructions": "Well?"}}
        post(self.client, state="shared doc", questions=questions)
        after_first = self.model.prefills
        post(self.client, state="shared doc", questions=questions)
        body = post(self.client, state="shared doc", questions=questions).json()
        self.assertEqual(self.model.prefills, after_first)
        self.assertGreater(body["usage"]["cached_input_tokens"], 0)

    def test_a_different_state_prefills_again(self):
        questions = {"q": {"type": "noul", "instructions": "Well?"}}
        post(self.client, state="doc one", questions=questions)
        before = self.model.prefills
        post(self.client, state="doc two", questions=questions)
        self.assertEqual(self.model.prefills, before + 1)

    def test_structured_state_is_accepted(self):
        response = post(
            self.client,
            state={"ticket": 42, "tags": ["billing", "urgent"]},
            questions={"q": {"type": "noul", "instructions": "Urgent?"}},
        )
        self.assertEqual(response.status_code, 200)

    def test_cache_reset_clears_the_states(self):
        post(self.client, questions={"q": {"type": "noul", "instructions": "Hm?"}})
        body = self.client.post("/v1/cache/reset").json()
        self.assertEqual(body["state_cache"]["entries"], 0)

    def test_rejects_a_request_with_no_questions(self):
        self.assertEqual(post(self.client, questions={}).status_code, 422)

    def test_rejects_a_choice_with_one_option(self):
        response = post(
            self.client,
            questions={
                "q": {"type": "choice", "instructions": "?", "criteria": {"only": None}}
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_rejects_more_options_than_single_token_labels(self):
        criteria = {f"option_{i}": None for i in range(30)}
        response = post(
            self.client,
            questions={"q": {"type": "choice", "instructions": "?", "criteria": criteria}},
        )
        self.assertEqual(response.status_code, 400)

    def test_reads_is_bounded(self):
        questions = {"q": {"type": "noul", "instructions": "?"}}
        self.assertEqual(post(self.client, reads=0, questions=questions).status_code, 422)
        self.assertEqual(post(self.client, reads=999, questions=questions).status_code, 422)

    def test_more_reads_report_a_spread(self):
        body = post(
            self.client,
            reads=6,
            questions={"q": {"type": "noul", "instructions": "?"}},
        ).json()
        self.assertIn("stderr", body["answers"]["q"])


class TestConfidence(unittest.TestCase):
    def test_uniform_is_zero_and_certain_is_one(self):
        self.assertAlmostEqual(_confidence([0.5, 0.5]), 0.0, places=6)
        self.assertAlmostEqual(_confidence([1.0, 0.0]), 1.0, places=6)

    def test_confidence_is_measured_against_the_uniform_baseline(self):
        # The same top probability is more decisive over more options, because
        # the baseline it has to beat is lower: 0.6 against a 0.33 coin-flip
        # says more than 0.6 against a 0.5 one.
        two = _confidence([0.6, 0.4])
        three = _confidence([0.6, 0.2, 0.2])
        self.assertGreater(three, two)

    def test_confidence_rises_as_the_winner_pulls_ahead(self):
        self.assertGreater(_confidence([0.9, 0.1]), _confidence([0.6, 0.4]))


class TestStateCache(unittest.TestCase):
    def test_evicts_the_least_recently_used(self):
        cache = StateCache(capacity=2)
        cache.put("a", "A")
        cache.put("b", "B")
        cache.get("a")
        cache.put("c", "C")
        self.assertIsNotNone(cache.get("a"))
        self.assertIsNotNone(cache.get("c"))
        self.assertIsNone(cache.get("b"))

    def test_tracks_hits_and_misses(self):
        cache = StateCache(capacity=2)
        cache.put("a", "A")
        cache.get("a")
        cache.get("missing")
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)


class TestQuestionCompilation(unittest.TestCase):
    def test_noul_compiles_to_two_labelled_options(self):
        compiled = compile_question(
            StubTokenizer(), "q", Question(type="noul", instructions="Urgent?")
        )
        self.assertEqual(compiled.kind, "noul")
        self.assertEqual(len(compiled.plan.choices), 2)

    def test_choice_preserves_option_order(self):
        compiled = compile_question(
            StubTokenizer(),
            "q",
            Question(
                type="choice",
                instructions="Which?",
                criteria={"first": None, "second": None, "third": None},
            ),
        )
        self.assertEqual(compiled.labels, ["first", "second", "third"])


if __name__ == "__main__":
    unittest.main()
