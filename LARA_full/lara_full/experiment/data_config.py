# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
from abc import ABC, abstractmethod
from typing import Sequence

from lara_full.data.dataset import ModalityConfig
from lara_full.data.transform.base import ComposedModalityTransform, ModalityTransform
from lara_full.data.transform.concat import ConcatTransform
from lara_full.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from lara_full.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from lara_full.model.transforms import GR00TTransform


class BaseDataConfig(ABC):
    _parse_attributes = ["observation_indices", "action_indices"]

    video_keys: Sequence[str] = []
    state_keys: Sequence[str] = []
    action_keys: Sequence[str] = []
    language_keys: Sequence[str] = []
    observation_indices: Sequence[int] = []
    action_indices: Sequence[int] = []
    video_backend: str = "decord"

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            if isinstance(value, str) and key in self._parse_attributes:
                value = ast.literal_eval(value)
            setattr(self, key, value)

    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


class OXEAustinBudsConfig(BaseDataConfig):
    video_keys = ["video.image"]
    state_keys = [
        "state.motors",
        "state.gripper",
    ]
    action_keys = [
        "action.position",
        "action.orientation",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    video_backend = "torchvision_av"

    state_normalization_modes = {
        "state.motors": "mean_std",
        "state.gripper": "min_max",
    }
    action_normalization_modes = {
        "action.position": "mean_std",
        "action.orientation": "mean_std",
        "action.gripper": "min_max",
    }

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self) -> ModalityTransform:
        return ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=self.video_keys),
                VideoCrop(apply_to=self.video_keys, scale=0.95),
                VideoResize(
                    apply_to=self.video_keys,
                    height=224,
                    width=224,
                    interpolation="linear",
                ),
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
                ),
                VideoToNumpy(apply_to=self.video_keys),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=self.state_normalization_modes,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=self.action_normalization_modes,
                ),
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
                GR00TTransform(
                    state_horizon=len(self.observation_indices),
                    action_horizon=len(self.action_indices),
                    max_state_dim=64,
                    max_action_dim=32,
                ),
            ]
        )


class LiberoDataConfig(OXEAustinBudsConfig):
    """LIBERO Panda observations and actions used for Stage 3 post-training."""

    video_keys = ["video.image", "video.wrist_image"]
    state_keys = [
        "state.position",
        "state.orientation",
        "state.gripper",
    ]
    action_keys = [
        "action.position",
        "action.orientation",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    video_backend = "torchvision_av"

    state_normalization_modes = {
        "state.position": "mean_std",
        "state.orientation": "mean_std",
        "state.gripper": "min_max",
    }
    action_normalization_modes = {
        "action.position": "mean_std",
        "action.orientation": "mean_std",
        "action.gripper": "min_max",
    }
