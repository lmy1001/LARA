"""Utilities for loading a released latent-motion-tokenizer checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf


def _load_config(checkpoint_dir: Path):
    yaml_path = checkpoint_dir / "config.yaml"
    json_path = checkpoint_dir / "config.json"

    if yaml_path.is_file():
        return OmegaConf.load(yaml_path)
    if json_path.is_file():
        with json_path.open(encoding="utf-8") as handle:
            return OmegaConf.create(json.load(handle))
    raise FileNotFoundError(
        f"Tokenizer checkpoint has no config.yaml or config.json: {checkpoint_dir}"
    )


def _load_state_dict(checkpoint_dir: Path) -> tuple[dict, Path]:
    safetensors_path = checkpoint_dir / "model.safetensors"
    pytorch_path = checkpoint_dir / "pytorch_model.bin"

    if safetensors_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu"), safetensors_path
    if pytorch_path.is_file():
        return torch.load(pytorch_path, map_location="cpu", weights_only=True), pytorch_path
    raise FileNotFoundError(
        f"Tokenizer checkpoint has no model.safetensors or pytorch_model.bin: {checkpoint_dir}"
    )


def load_latent_motion_tokenizer(
    checkpoint_path: str | Path,
    *,
    image_encoder_path: str | Path | None = None,
):
    """Instantiate and restore a tokenizer without relying on repository-local paths."""

    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer checkpoint directory does not exist: {checkpoint_dir}")

    config = _load_config(checkpoint_dir)
    if image_encoder_path is not None:
        encoder_path = str(Path(image_encoder_path).expanduser())
        encoder_config_key = (
            "config.image_encoder_config.pretrained_model_name_or_path"
            if "config" in config
            else "image_encoder_config.pretrained_model_name_or_path"
        )
        OmegaConf.update(
            config,
            encoder_config_key,
            encoder_path,
            merge=False,
            force_add=True,
        )

    state_dict, weights_path = _load_state_dict(checkpoint_dir)
    if "_target_" in config:
        model = hydra.utils.instantiate(config)
    else:
        from moto.latent_motion_tokenizer.configs import LatentMotionTokenizerConfig
        from moto.latent_motion_tokenizer.model.latent_motion_tokenizer import (
            LatentMotionTokenizer,
        )

        config_dict = OmegaConf.to_container(config, resolve=True)
        model = LatentMotionTokenizer(LatentMotionTokenizerConfig(**config_dict))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    ignored_missing_roots = {"image_encoder", "loss_fn_lpips"}
    missing_roots = {key.split(".", 1)[0] for key in missing_keys} - ignored_missing_roots
    if missing_roots or unexpected_keys:
        print(
            "Loaded tokenizer with non-strict keys:",
            {
                "weights": str(weights_path),
                "missing_roots": sorted(missing_roots),
                "unexpected_keys": sorted(unexpected_keys),
            },
        )
    return model
