# System One server

A decision endpoint for masked-diffusion models, following TypeSafe's
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) interface:
one *state* plus any number of typed *questions*, answered in a single call with
calibrated probabilities. No token generation, no output parsing.

```bash
uv run python -m mlx_vlm.systemone \
  --model google/diffusiongemma-26B-A4B-it \
  --trust-remote-code --port 8100
```

```bash
curl -s localhost:8100/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I have been trying to connect my Stripe account for 3 days and nothing works.
            I have emailed twice with no reply. This is blocking our launch.",
  "questions": {
    "urgency":  {"type": "noul",   "instructions": "Does this message express urgency?"},
    "category": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "payment issues",
                              "integrations": "third-party connections",
                              "sales": "new purchase enquiries"}}
  }
}'
```

```json
{
  "answers": {
    "urgency":  {"type": "noul", "noul": 0.993, "confidence": 0.937, "stderr": 0.001},
    "category": {"type": "choice", "choice": "integrations",
                 "probabilities": {"billing": 0.056, "integrations": 0.901, "sales": 0.043},
                 "confidence": 0.645, "stderr": 0.023}
  },
  "usage": {"input_tokens": 1184, "output_tokens": 5,
            "cached_input_tokens": 0, "forward_passes": 1}
}
```

## How it works

Each question is compiled into a denoising canvas seeded with its text, leaving
a single free slot for the answer. Options are labelled `A`, `B`, `C`… because a
slot holds exactly one token. Every question — and every repeat — is stacked into
**one batched forward pass**, so `forward_passes` is 1 whether the request
carries one question or twenty.

States are cached across requests by content hash. Asking a second batch of
questions about a document already seen skips the prefill, which is where most
of a read's cost sits: on a 1833-token state that is 22.9s → 2.1s for ten reads.

## Primitives

| type | criteria | answer |
|---|---|---|
| `noul` | `{"true": "...", "false": "..."}` (optional) | `noul`: P(yes) |
| `choice` | `{option: description}` | `choice`, `probabilities`, `confidence` |
| `score` | `["level_0", "level_1", ...]` | `score`, `mode`, `legend`, `probabilities` |

`reads` (default 4) averages independent canvases and reports the spread as
`stderr`. One read is fastest; more is steadier on close calls.

`confidence` is `1 - H(p)/log(n)`, i.e. distance from uniform. This is our own
definition — TypeSafe does not publish theirs. Note it is not the top
probability: 0.6 over three options scores higher than 0.6 over two, because the
baseline it has to beat is lower.

## Measured on `diffusiongemma-26B-A4B-it` (MXFP4)

| | |
|---|---|
| single `noul`, `reads=1` | ~250 ms |
| 5 questions × 8 reads, cold state | 5.3s |
| same, cached state | 1.3s |

## Limits worth knowing

**`score` is the weak primitive.** Three levels behave; five go bimodal, putting
mass at both ends of the scale so the expected value lands in a middle the model
never chose. On "production database is corrupted" a 5-level severity read gave
`score=2.41` with `mode=4 (critical)` at `confidence=0.23`. That is why `mode` is
reported alongside `score`: **when the two disagree, or confidence is low, treat
the reading as unreliable.** Prefer `noul` or `choice`, or keep scales to three
levels.

**Canvases pad to the widest question in a request.** Long instructions make the
whole batch wider, so group similar sizes rather than mixing a short question
with a paragraph-long one.

**Requests are served sequentially.** The batching is within a request, across
its questions — not across concurrent HTTP clients.

**Options are capped at 26** (single-token labels `A`–`Z`), and far fewer in
practice before a read stops separating them.
