from __future__ import annotations

import argparse
import asyncio
import json
import os

from .orchestrator import PipelineOrchestrator


async def run_benchmark(frames: int, target_fps: int) -> dict[str, object]:
    original_transport = os.getenv("PIPELINE_TRANSPORT")
    results: dict[str, object] = {}
    try:
        os.environ["PIPELINE_TRANSPORT"] = "http"
        http_orchestrator = PipelineOrchestrator(frames=frames, target_fps=target_fps)
        await http_orchestrator.run()
        results["http"] = http_orchestrator.snapshot().model_dump()

        os.environ["PIPELINE_TRANSPORT"] = "queue"
        queue_orchestrator = PipelineOrchestrator(frames=frames, target_fps=target_fps)
        await queue_orchestrator.run()
        results["queue"] = queue_orchestrator.snapshot().model_dump()
    finally:
        if original_transport is None:
            os.environ.pop("PIPELINE_TRANSPORT", None)
        else:
            os.environ["PIPELINE_TRANSPORT"] = original_transport
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HTTP vs queue pipeline modes")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--target-fps", type=int, default=12)
    args = parser.parse_args()
    results = asyncio.run(run_benchmark(args.frames, args.target_fps))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
