# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Edge visual generation on one CUDA GPU; set RUN_COSMOS3_EDGE_TESTS=1."""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest
from sglang_omni.models.cosmos3.config import Cosmos3PipelineConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.utils.checkpoint import resolve_checkpoint

pytestmark = [pytest.mark.benchmark, pytest.mark.accelerator]

EDGE = "nvidia/Cosmos3-Edge@a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba"
CASES = (
    "t2i_landscape",
    "t2i_portrait",
    "t2v_53",
    "t2v_121",
    "i2v_53",
    "i2v_121",
    "video_continuation_53",
)
Records = dict[str, dict[str, Any]]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def build_requests(checkpoint: Path, directory: Path) -> Records:
    assets = checkpoint / "assets"
    prompt = json.dumps(json.loads((assets / "example_i2v_prompt.json").read_text()))
    negative = json.dumps(json.loads((assets / "negative_prompt.json").read_text()))
    cases = {}
    for name in CASES:
        large = name.endswith("121")
        width, height = (832, 480) if large else (448, 256)
        frames = 121 if large else 53
        text = prompt
        if name.startswith("t2i"):
            frames = 1
            text = "A red ceramic mug stands on a wooden table in daylight."
            if name.endswith("portrait"):
                width, height = 256, 448
                text = "A tall green plant in a white pot beside a window."
        params = {
            "prompt": text,
            "negative_prompt": negative,
            "width": width,
            "height": height,
            "num_frames": frames,
            "fps": 24,
            "num_inference_steps": 20,
            "guidance_scale": 6.0,
            "flow_shift": 12.0,
            "seed": 0,
            "use_system_prompt": False,
            "use_resolution_template": False,
            "use_duration_template": False,
            "use_guardrails": False,
            "sound_duration": 0.0,
            "output_compression": 100,
        }
        if name.startswith("i2v"):
            image_path = directory / f"conditioning_{width}x{height}.png"
            with Image.open(assets / "example_i2v_input.jpg") as image:
                ImageOps.fit(
                    image.convert("RGB"),
                    (width, height),
                    method=Image.Resampling.BILINEAR,
                ).save(image_path)
            params["image_path"] = str(image_path)
        elif name.startswith("video_continuation"):
            params.update(
                video_path=str(assets / "example_action_id_av_0_input.mp4"),
                condition_video_keep="first",
            )
        cases[name] = params
    return cases


async def run_arm(
    checkpoint: Path, directory: Path, requests: Records, optimized: bool
) -> Records:
    config = Cosmos3PipelineConfig(model_path=str(checkpoint))
    options = {"attention_backend": "torch_sdpa", "warmup_mode": "off"}
    if optimized:
        options.update(
            component_residency={"transformer": "layerwise-offload"},
            layerwise_resident_layers={"transformer": 32},
        )
    config.stages[0].factory.server_args_overrides = options
    config.stages[0].factory.output_dir = str(directory / "media")
    directory.mkdir()
    write_json(directory / "config.json", config.model_dump(mode="json"))
    runner = MultiProcessPipelineRunner(config)
    records = {}
    try:
        await runner.start(timeout=900)
        client = Client(runner.coordinator)
        for name, params in requests.items():
            started = time.perf_counter()
            result = await asyncio.wait_for(
                client.completion(
                    GenerateRequest(prompt=params, stream=False), request_id=name
                ),
                timeout=900,
            )
            elapsed = time.perf_counter() - started
            assert result.finish_reason == "stop"
            assert result.media and len(result.media) == 1
            path = Path(result.media[0]["path"])
            pixels = iio.imread(path)
            if pixels.ndim == 3:
                pixels = pixels[None]
            assert pixels.shape == (
                params["num_frames"],
                params["height"],
                params["width"],
                3,
            )
            assert pixels.dtype == np.uint8 and pixels.std() > 1
            records[name] = {
                "path": str(path),
                "shape": list(pixels.shape),
                "sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "request_wall_s": elapsed,
                "result": asdict(result),
            }
            write_json(directory / f"{name}.json", records[name])
    finally:
        await asyncio.wait_for(runner.stop(), timeout=90)
    for record in records.values():
        assert Path(record["path"]).is_file(), "Successful output lost on shutdown"
    return records


@pytest.fixture(scope="module")
def visual_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Records, Records]:
    if os.environ.get("RUN_COSMOS3_EDGE_TESTS") != "1":
        pytest.skip("Set RUN_COSMOS3_EDGE_TESTS=1 to load the official Edge checkpoint")
    if not torch.cuda.is_available():
        pytest.skip("Cosmos3 Edge generation requires CUDA")
    checkpoint = Path(resolve_checkpoint(os.environ.get("COSMOS3_EDGE_MODEL", EDGE)))
    directory = Path(
        os.environ.get("COSMOS3_TEST_OUTPUT")
        or tmp_path_factory.mktemp("cosmos3-edge-generation")
    ).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    requests = build_requests(checkpoint, directory)
    write_json(directory / "requests.json", requests)
    baseline = asyncio.run(run_arm(checkpoint, directory / "baseline", requests, False))
    optimized = asyncio.run(
        run_arm(checkpoint, directory / "optimized", requests, True)
    )
    return baseline, optimized


@pytest.mark.parametrize("case", CASES)
def test_visual_residency_preserves_decoded_pixels(
    visual_runs: tuple[Records, Records], case: str
) -> None:
    """Layerwise offload must not change the decoded pixels of any visual task."""
    baseline, optimized = visual_runs
    assert optimized[case]["sha256"] == baseline[case]["sha256"]
