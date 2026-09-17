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
| `score` | `["level_0", "level_1", ...]` (max 10) | `score`, `mode`, `bimodal`, `legend`, `probabilities` |

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

**`score` labels its levels with digits, not letters.** Letters carry no order,
so a scored read on `A`–`E` scatters mass across both ends of the scale and the
expected value lands in a middle the model never chose. Digits are ordinal and
the model reads them as the scale they are. Over a 5-level severity set with
known answers:

| labels | mean abs. error | modal level correct | monotonic |
|---|---|---|---|
| `A`–`E` | 1.01 | 20% | no |
| `1`–`5` | 0.57 | 80% | no |
| `0`–`4` (index-aligned) | **0.25** | 60% | **yes** |

On "production database corrupted" the distribution goes from
`[0.21 0.23 0.02 0.08 0.47]` (bimodal, `score=2.35`) to
`[0.05 0.03 0.01 0.02 0.90]` (`score=3.70`, `mode=4`).

`score` is still the primitive to watch. Scores run monotonic across a severity
ladder but drift about half a level high in the middle of the range, and a
`bimodal: true` flag or a low `confidence` means the expected value is not a
summary to act on — check `mode` instead. On the one case above where the modal
level was wrong, confidence read 0.09. Prefer `noul` or `choice` when a decision
has to be right.

**Canvases pad to the widest question in a request.** Long instructions make the
whole batch wider, so group similar sizes rather than mixing a short question
with a paragraph-long one.

**Requests are served sequentially.** The batching is within a request, across
its questions — not across concurrent HTTP clients.

**Options are capped at 26** (single-token labels `A`–`Z`), and far fewer in
practice before a read stops separating them.

## Images

Pass `images` alongside (or instead of) `state`. Each entry is a data URL, an
http(s) URL, or a local path. The image is encoded into the same cached prefix
as text, so questions about a picture cost no more than questions about a
document, and the images take part in the cache key.

```bash
curl -s localhost:8100/v1/systemone -H 'Content-Type: application/json' -d '{
  "images": ["data:image/png;base64,iVBORw0KGgo..."],
  "questions": {
    "red":   {"type": "noul",   "instructions": "Is this image mostly red?"},
    "shape": {"type": "choice", "instructions": "What shape is in this image?",
              "criteria": {"circle": null, "square": null, "triangle": null}}
  }
}'
```

Requires a checkpoint whose vision tower is not float16 — see
`_vision_needs_promotion` in the model for why some repacks silently emit NaN
for every image.

## Clients

`examples/systemone_client.mjs` is a dependency-free Node 18+ client covering
text, structured state, state reuse and images:

```bash
node examples/systemone_client.mjs                # text examples
node examples/systemone_client.mjs photo.png      # adds the image example
```

## Ask atomic questions, compose in code

`examples/systemone_eval_agent.mjs` grades an agent trace with three planted
faults: a refund above the approval threshold with no approval_token, a
replacement promised for a SKU the agent's own `check_stock` reported
unavailable, and a missing prepaid return label. The customer ends delighted.

Asked directly, the composite judgments come back **confidently wrong**:

```
issue_resolved        88.6%   conf 0.49     (the replacement never happened)
factually_consistent  89.6%   conf 0.52     (the agent contradicted check_stock)
policy_adherence      82.5%   conf 0.33     (refund had no approval_token)
```

A single denoising step answers from the surface of the trace — happy customer,
closed case — rather than by chaining the facts that contradict it. Decomposed,
each answer is sharp:

```
stock_available        0.8%   conf 0.93
approval_token_used    7.8%   conf 0.60
refund_over_threshold 85.2%   conf 0.39
promised_replacement  88.7%   conf 0.49
```

Compose those in code and all three faults are found, stably across runs:

```js
if (yes(a.refund_issued) && yes(a.refund_over_threshold) && !yes(a.approval_token_used))
  findings.push("refund above threshold issued without supervisor approval");
```

This is the shape to reach for. Questions of the form "is X true of this state?"
work; "did everything go well?" does not, and will not tell you it has failed.
