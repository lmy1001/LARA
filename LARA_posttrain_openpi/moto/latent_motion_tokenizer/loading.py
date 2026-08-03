"""Utilities for loading a released latent-motion-tokenizer checkpoint."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch

_OPTIONAL_CHECKPOINT_ROOTS = {"image_encoder", "loss_fn_lpips"}
_STATE_DICT_PREFIXES = ("module.", "model.", "latent_motion_tokenizer.")


def _load_config(checkpoint_dir: Path):
    yaml_path = checkpoint_dir / "config.yaml"
    json_path = checkpoint_dir / "config.json"

    if yaml_path.is_file():
        return OmegaConf.load(yaml_path)
    if json_path.is_file():
        with json_path.open(encoding="utf-8") as handle:
            return OmegaConf.create(json.load(handle))
    raise FileNotFoundError(f"Tokenizer checkpoint has no config.yaml or config.json: {checkpoint_dir}")


def _load_state_dict(checkpoint_dir: Path) -> tuple[dict, Path]:
    safetensors_path = checkpoint_dir / "model.safetensors"
    pytorch_path = checkpoint_dir / "pytorch_model.bin"

    if safetensors_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu"), safetensors_path
    if pytorch_path.is_file():
        return torch.load(pytorch_path, map_location="cpu", weights_only=True), pytorch_path
    raise FileNotFoundError(f"Tokenizer checkpoint has no model.safetensors or pytorch_model.bin: {checkpoint_dir}")


def _normalize_state_dict(state_dict: dict) -> dict[str, torch.Tensor]:
    """Unwrap common checkpoint containers and remove uniform wrapper prefixes."""

    for container_key in ("state_dict", "model"):
        nested = state_dict.get(container_key)
        if isinstance(nested, dict):
            state_dict = nested
            break

    if not state_dict or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state_dict.items()
    ):
        raise ValueError("Tokenizer checkpoint must contain a non-empty tensor state dict.")

    normalized = dict(state_dict)
    prefix_removed = True
    while prefix_removed:
        prefix_removed = False
        for prefix in _STATE_DICT_PREFIXES:
            if all(key.startswith(prefix) for key in normalized):
                normalized = {key.removeprefix(prefix): value for key, value in normalized.items()}
                prefix_removed = True
                break
    return normalized


def load_latent_motion_tokenizer(
    checkpoint_path: str | Path,
    *,
    image_encoder_path: str | Path | None = None,
):
    """Instantiate and restore a tokenizer without repository-local paths."""

    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer checkpoint directory does not exist: {checkpoint_dir}")

    config = _load_config(checkpoint_dir)
    if image_encoder_path is not None:
        encoder_path = Path(image_encoder_path).expanduser()
        if not encoder_path.exists():
            raise FileNotFoundError(f"Tokenizer image encoder does not exist: {encoder_path}")
        encoder_config_key = (
            "config.image_encoder_config.pretrained_model_name_or_path"
            if "config" in config
            else "image_encoder_config.pretrained_model_name_or_path"
        )
        OmegaConf.update(config, encoder_config_key, str(encoder_path), merge=False, force_add=True)

    state_dict, weights_path = _load_state_dict(checkpoint_dir)
    state_dict = _normalize_state_dict(state_dict)
    if "_target_" in config:
        model = hydra.utils.instantiate(config)
    else:
        from moto.latent_motion_tokenizer.configs import LatentMotionTokenizerConfig
        from moto.latent_motion_tokenizer.model.latent_motion_tokenizer import LatentMotionTokenizer

        config_dict = OmegaConf.to_container(config, resolve=True)
        model = LatentMotionTokenizer(LatentMotionTokenizerConfig(**config_dict))

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    required_missing = [key for key in missing_keys if key.split(".", 1)[0] not in _OPTIONAL_CHECKPOINT_ROOTS]
    required_unexpected = [key for key in unexpected_keys if key.split(".", 1)[0] not in _OPTIONAL_CHECKPOINT_ROOTS]
    if required_missing or required_unexpected:
        raise RuntimeError(
            f"Tokenizer checkpoint is incompatible with the release model: {weights_path}; "
            f"missing={required_missing[:20]}, unexpected={required_unexpected[:20]}"
        )
    if missing_keys or unexpected_keys:
        logging.info(
            "Tokenizer checkpoint %s omits external frozen modules (%d missing, %d unexpected keys).",
            weights_path,
            len(missing_keys),
            len(unexpected_keys),
        )
    return model
