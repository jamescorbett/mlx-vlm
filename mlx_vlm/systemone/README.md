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

Only one server may run at a time: the model is tens of gigabytes, and a second
copy gets one of them OOM-killed mid-request. A second launch exits immediately
with the holder's pid rather than competing for memory:

```
A System One server is already running on port 8100 (pid 54856).
Stop it with:  kill 54856
Lock file:     /var/folders/.../mlx_vlm_systemone.lock
```

Locks left by a dead process are reclaimed automatically. `--no-lock` opts out.

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

## Composite questions need a reasoning pass

`examples/systemone_eval_agent.mjs` grades an agent trace with three planted
faults: a $408 refund against a $200 approval threshold with no approval_token,
a replacement promised for a SKU the agent's own `check_stock` reported
unavailable, and a missing prepaid return label. The customer ends delighted, so
the surface of the trace disagrees with its substance.

Read directly, this server answers from the surface and is confidently wrong.
Set `"reasoning": true` and it matches TypeSafe's hosted Jev:

| question | truth | Jev | direct | `reasoning: true` |
|---|---|---|---|---|
| `factually_consistent` | no | 0.04 | 0.69 | **0.00** |
| `policy_adherence` | no | 0.03 | 0.57 | **0.00** |
| `issue_resolved` | no | 0.35 | 0.95 | **0.16** |
| `escalation_needed` | yes | 0.92 | 0.13 | **0.83** |
| | | 4/4 | 0/4 | **4/4** |

4/4 on three consecutive runs. The cost is one generation pass: roughly 6s to
25s on this checkpoint, so leave it off for surface questions, which do not
need it and are far faster without.

So the limit was never precision. The same 4-bit model finds the faults
perfectly well — it just cannot do it *inside a single denoising step*, where
there is nowhere to hold an intermediate conclusion. Given somewhere to write
one down, it gets there.

### The analysis prompt has to be adversarial

This is the part that decides whether the pass helps at all. Asked neutrally to
"state the specific facts that decide each check", the model writes a defence:

> **Policy Compliance:** The agent issued a refund for a damaged item, which is
> permitted under the `get_policy` result

— never comparing $408 to the $200 threshold sitting a few lines above. That
phrasing scored **0/4**, worse than not reasoning at all. Asked for mistakes and
contradictions, the same model produces:

> **Contradiction/Error:** The agent claimed to have "arranged a replacement
> espresso machine to arrive before Saturday," but the `check_stock` tool
> (seq 4) showed the Presto machine (ESP-900) was out of stock and backordered

The notes come back on the response as `reasoning`, so a decision can be
audited rather than trusted.

### Or decompose instead

Without a reasoning pass, atomic questions still work and stay cheap:

```
stock_available        0.8%   conf 0.93
approval_token_used    7.8%   conf 0.60
refund_over_threshold 85.2%   conf 0.39
```

```js
if (yes(a.refund_issued) && yes(a.refund_over_threshold) && !yes(a.approval_token_used))
  findings.push("refund above threshold issued without supervisor approval");
```

That finds all three faults, identically across runs, in one forward pass.
Naming the evidence inside a composite question helps too — rewording them to
point at the specific fields took 1/4 to 3/4 with no extra compute.

### What does not work: reasoning inside the canvas

The canvas attends to itself bidirectionally, so it is tempting to put the
intermediate steps in it alongside the answer and skip the generation pass:

```
Facts: the refund exceeded the approval threshold: @.
An approval token was included: @.
Therefore the agent complied with policy: @
```

Measured 0/3 — no better than asking the composite question directly. The slots
denoise simultaneously and correlate, so the canvas collapses to one repeated
letter instead of stepping through:

```
refund exceeded the threshold   0.93 yes   (correct)
approval token was included     0.96 yes   (wrong — none was passed)
therefore complied              0.98 yes   (wrong)
```

The middle slot reads 0.078 when asked as its own separate read. Independence is
what makes decomposition work, and putting the steps in one canvas destroys it.

### One point of agreement with Jev

On `predicted_csat` Jev returns a split distribution —
`[0.12, 0.43, 0.02, 0.06, 0.37]`, mass at both "Dissatisfied" and "Very
satisfied" — and reports `confidence: 0`. This server's `bimodal` flag fires on
exactly that shape, which is the signal that an expected value should not be
acted on.
