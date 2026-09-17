"""A System One server for masked-diffusion models.

One *state* plus any number of typed *questions*, answered in a single call.
The questions ride in the denoising canvas rather than the prompt, so a request
costs one prefill of the state and one batched forward pass over all its
questions, however many there are.

States are cached across requests: asking a second batch of questions about a
document already seen skips the prefill entirely, which is where most of the
cost of a read sits.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import tempfile
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import mlx.core as mx
from fastapi import FastAPI, HTTPException

from ..prompt_utils import apply_chat_template
from ..structured_reads import ReadSession, _summarize
from ..utils import prepare_inputs
from .decisions import build_answer, compile_question, render
from .schemas import SystemOneRequest, SystemOneResponse, Usage

logger = logging.getLogger("mlx_vlm.systemone")

DEFAULT_STATE_CACHE_SIZE = 32


def _materialize_image(image: str) -> str:
    """Accept a data URL, an http(s) URL, or a path, and hand back a loadable one.

    Data URLs are the only form a browser can produce without hosting the file
    somewhere, so they are decoded to a temporary file for the image loader.
    """
    if not image.startswith("data:"):
        return image
    try:
        header, _, payload = image.partition(",")
        if not payload:
            raise ValueError("data URL has no payload")
        suffix = ".png"
        if "/" in header:
            mime = header.split(";")[0].split("/")[-1]
            if mime.isalnum():
                suffix = f".{mime.replace('jpeg', 'jpg')}"
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail=f"Malformed data URL: {exc}") from exc

    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    handle.write(raw)
    handle.close()
    return handle.name


class StateCache:
    """LRU of prefilled states, keyed by the rendered state text."""

    def __init__(self, capacity: int = DEFAULT_STATE_CACHE_SIZE):
        self.capacity = max(1, capacity)
        self._entries: "OrderedDict[str, ReadSession]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[ReadSession]:
        session = self._entries.get(key)
        if session is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return session

    def put(self, key: str, session: ReadSession) -> None:
        self._entries[key] = session
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._entries),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
        }


class SystemOneRuntime:
    def __init__(self, model, processor, model_id: str, cache_size: int):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.model_id = model_id
        self.states = StateCache(cache_size)

    def session_for(
        self, state_text: str, images: Optional[List[str]] = None
    ) -> tuple[ReadSession, bool, int]:
        images = list(images or [])
        key = StateCache.key(state_text + "\x00" + "\x00".join(images))
        cached = self.states.get(key)
        if cached is not None:
            return cached, True, cached.prompt_tokens

        if images:
            resolved = [_materialize_image(image) for image in images]
            instruction = "Look at this and answer questions about it."
            if state_text:
                instruction += f"\n\n{state_text}"
            prompt = apply_chat_template(
                self.processor, self.model.config, instruction, num_images=len(resolved)
            )
            inputs = prepare_inputs(
                self.processor,
                images=resolved,
                prompts=[prompt],
                image_token_index=getattr(self.model.config, "image_token_id", None),
            )
            session = ReadSession(
                self.model,
                self.processor,
                self.tokenizer,
                inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
            )
        else:
            prompt = apply_chat_template(
                self.processor,
                self.model.config,
                f"Read this state and answer questions about it.\n\n{state_text}",
            )
            input_ids = mx.array([self.tokenizer.encode(prompt)])
            session = ReadSession(self.model, self.processor, self.tokenizer, input_ids)

        self.states.put(key, session)
        return session, False, session.prompt_tokens


def create_app(model, processor, model_id: str, cache_size: int = DEFAULT_STATE_CACHE_SIZE) -> FastAPI:
    runtime = SystemOneRuntime(model, processor, model_id, cache_size)
    app = FastAPI(title="mlx-vlm System One", version="1.0")
    app.state.runtime = runtime

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model": runtime.model_id,
            "state_cache": runtime.states.stats(),
        }

    @app.post("/v1/cache/reset")
    async def reset_cache():
        runtime.states.clear()
        return {"status": "ok", "state_cache": runtime.states.stats()}

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def systemone(body: SystemOneRequest):
        started = time.perf_counter()
        state_text = render(body.state)

        try:
            compiled = [
                compile_question(runtime.tokenizer, key, question)
                for key, question in body.questions.items()
            ]
        except ValueError as exc:
            # A question that cannot be reduced to one canvas slot is a client
            # error, and saying so beats returning a confidently wrong reading.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not state_text and not body.images:
            raise HTTPException(
                status_code=400, detail="a request needs a state, images, or both"
            )

        session, cache_hit, prompt_tokens = runtime.session_for(state_text, body.images)

        # Every question, every repeat, in one pass.
        width = max(item.plan.width for item in compiled)
        plans = [item.plan for item in compiled for _ in range(body.reads)]
        results = session.read_batch(plans, canvas_length=width)

        answers: Dict[str, Dict[str, Any]] = {}
        for index, item in enumerate(compiled):
            window = results[index * body.reads : (index + 1) * body.reads]
            decision = _summarize(item.plan, window)
            probabilities = [
                decision.probabilities[choice] for choice in item.plan.choices
            ]
            answers[item.key] = build_answer(item, probabilities, decision.stderr)

        elapsed = time.perf_counter() - started
        logger.info(
            "systemone: %d questions x%d reads in %.0f ms (state %s, %d tokens)",
            len(compiled), body.reads, elapsed * 1000,
            "cached" if cache_hit else "prefilled", prompt_tokens,
        )

        return SystemOneResponse(
            model=runtime.model_id,
            answers=answers,
            usage=Usage(
                input_tokens=prompt_tokens + len(plans) * width,
                output_tokens=len(compiled),
                cached_input_tokens=prompt_tokens if cache_hit else 0,
                forward_passes=1,
            ),
        )

    return app
