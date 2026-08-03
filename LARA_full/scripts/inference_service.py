"""Serve a Stage 3 LARA/GR00T N1.5 checkpoint for simulator evaluation."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lara_full.eval.libero_config import LiberoDataConfig  # noqa: E402
from lara_full.eval.service import RobotInferenceServer  # noqa: E402
from lara_full.model.policy import Gr00tPolicy  # noqa: E402


@dataclass
class ServerConfig:
    model_path: str
    host: str = "*"
    port: int = 5555
    denoising_steps: int = 4
    device: str = "cuda"


def _validate_checkpoint(model_path: str) -> Path:
    checkpoint = Path(model_path).expanduser().resolve()
    config_path = checkpoint / "config.json"
    metadata_path = checkpoint / "experiment_cfg" / "metadata.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata: {metadata_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures", [])
    if config.get("model_type") != "gr00t_n1_5" or "GR00T_N1_5" not in architectures:
        raise ValueError(
            "This evaluator requires a GR00T N1.5 checkpoint "
            "(model_type=gr00t_n1_5, architecture=GR00T_N1_5)."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "libero" not in metadata:
        raise ValueError(f"No 'libero' normalization metadata found in {metadata_path}")
    return checkpoint


def main(config: ServerConfig) -> None:
    checkpoint = _validate_checkpoint(config.model_path)
    data_config = LiberoDataConfig()
    policy = Gr00tPolicy(
        model_path=str(checkpoint),
        embodiment_tag="libero",
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=config.denoising_steps,
        device=config.device,
    )
    RobotInferenceServer(
        policy=policy,
        host=config.host,
        port=config.port,
    ).run()


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
