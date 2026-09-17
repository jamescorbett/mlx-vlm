"""Run the System One server.

    uv run python -m mlx_vlm.systemone --model <path-or-repo> --port 8100
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from ..utils import load
from .app import DEFAULT_STATE_CACHE_SIZE, create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Diffusion model path or repo id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--state-cache-size",
        type=int,
        default=DEFAULT_STATE_CACHE_SIZE,
        help="Prefilled states kept for reuse across requests",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("mlx_vlm.systemone").setLevel(args.log_level)

    logging.info("Loading %s ...", args.model)
    model, processor = load(args.model, trust_remote_code=args.trust_remote_code)
    if getattr(model.config, "canvas_length", None) is None:
        raise SystemExit(
            f"{args.model} is not a masked-diffusion model; System One reads need "
            "a denoising canvas."
        )
    logging.info("Ready. Canvas length %d.", model.config.canvas_length)

    uvicorn.run(
        create_app(model, processor, args.model, args.state_cache_size),
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
