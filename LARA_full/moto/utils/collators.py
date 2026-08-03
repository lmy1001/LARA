"""Batch collation for latent-motion-tokenizer pretraining."""

from typing import Any

import numpy as np
import torch
from transformers.data.data_collator import DataCollatorMixin


def collate_moto_tokenizer(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not features or any("video" not in feature for feature in features):
        raise ValueError("Tokenizer samples must all contain a video field")

    video = torch.from_numpy(np.stack([sample["video"] for sample in features]))
    if video.ndim != 6 or video.shape[1] != 2 or video.shape[2] != 1:
        raise ValueError(
            "Expected video shape [batch, 2 frames, 1 camera, height, width, channels], "
            f"got {tuple(video.shape)}"
        )

    video = video.permute(0, 1, 2, 5, 3, 4).to(torch.float32) / 255.0
    return {
        "rgb_initial": video[:, 0],
        "rgb_future": video[:, 1],
    }


class DefaultDataCollatorMotoTokenizer(DataCollatorMixin):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return collate_moto_tokenizer(features)
