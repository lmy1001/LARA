"""Shared model and dataset factories for LARA training entry points."""

from __future__ import annotations

from pathlib import Path

import hydra
import omegaconf
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from lara_full.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from lara_full.data.schema import EmbodimentTag
from lara_full.experiment.data_config import BaseDataConfig


def model_factory(run_config: omegaconf.DictConfig) -> nn.Module:
    """Instantiate a model or restore it from a released checkpoint."""

    if not run_config.load_pretrained:
        return hydra.utils.instantiate(run_config.model)

    pretrained_path = str(run_config.pretrained_path)
    if not pretrained_path:
        raise ValueError("pretrained_path must be set when load_pretrained=true")

    if run_config.load_using_pretrained:
        if ".from_pretrained" in run_config.model._target_:
            raise ValueError("Model target already calls from_pretrained")
        model_config = omegaconf.OmegaConf.create(
            omegaconf.OmegaConf.to_container(run_config.model, resolve=False)
        )
        model_config._target_ = f"{model_config._target_}.from_pretrained"
        omegaconf.OmegaConf.update(
            model_config,
            "pretrained_model_name_or_path",
            pretrained_path,
            force_add=True,
        )
        return hydra.utils.instantiate(model_config)

    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    state_dict_key = run_config.get("pretrained_state_dict_key")
    state_dict = checkpoint[state_dict_key] if state_dict_key else checkpoint
    model = hydra.utils.instantiate(run_config.model)
    model.load_state_dict(state_dict)
    return model


def _get_single_data_config(
    single_data_config: omegaconf.DictConfig,
    *,
    train: bool,
) -> tuple[BaseDataConfig, object, object]:
    data_config: BaseDataConfig = hydra.utils.instantiate(single_data_config)
    modality_configs = data_config.modality_config()
    transforms = data_config.transform()
    if not train:
        transforms.eval()
    return data_config, modality_configs, transforms


def _iter_data_sections(data_config: omegaconf.DictConfig):
    if "data_config" in data_config:
        nested = data_config.data_config
        if "_target_" in nested:
            return [nested]
        return [nested[key] for key in nested]
    return [section.data_config for section in data_config.values()]


def dataset_factory(run_config: omegaconf.DictConfig, train: bool = True) -> Dataset:
    """Build one dataset or an equal-weight multi-embodiment mixture."""

    datasets: list[LeRobotSingleDataset] = []
    for section in _iter_data_sections(run_config.data):
        data_config, modality_configs, transforms = _get_single_data_config(
            section,
            train=train,
        )
        dataset_path = Path(str(data_config.dataset_path)).expanduser()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        datasets.append(
            LeRobotSingleDataset(
                dataset_path=str(dataset_path),
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=EmbodimentTag(data_config.embodiment_tag),
                video_backend=data_config.video_backend,
            )
        )

    if not datasets:
        raise ValueError("No dataset entries were configured")
    if len(datasets) == 1:
        return datasets[0]

    return LeRobotMixtureDataset(
        data_mixture=[(dataset, 1.0) for dataset in datasets],
        mode="train" if train else "eval",
        balance_dataset_weights=bool(run_config.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(
            run_config.get("balance_trajectory_weights", True)
        ),
        seed=int(run_config.get("seed", 42)),
        metadata_config={"percentile_mixing_method": "weighted_average"},
    )
