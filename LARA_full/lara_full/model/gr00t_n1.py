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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


def _to_plain_config(value):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_plain_config(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _to_plain_config(item) for key, item in value.items()}
    return value


def reset_parameters(module):
    for layer in module.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs, inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", False)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        use_latent_motion_queries = kwargs.pop("use_latent_motion_queries", False)
        latent_motion_tokenizer_path = kwargs.pop(
            "latent_motion_tokenizer_path", None
        )
        latent_motion_image_encoder_path = kwargs.pop(
            "latent_motion_image_encoder_path", None
        )
        tune_latent_motion_tokenizer = kwargs.pop(
            "tune_latent_motion_tokenizer", False
        )
        latent_motion_token_count = kwargs.pop("latent_motion_token_count", 8)
        latent_motion_codebook_dim = kwargs.pop("latent_motion_codebook_dim", 32)
        latent_motion_hidden_layer = kwargs.pop("latent_motion_hidden_layer", -3)
        latent_loss_weight = kwargs.pop("latent_loss_weight", 0.01)
        tokenizer_vae_loss_weight = kwargs.pop("tokenizer_vae_loss_weight", 0.01)
        reinitialize_action_head = kwargs.pop("reinitialize_action_head", False)

        print(f"Loading LARA base model from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        requested_path = Path(pretrained_model_name_or_path).expanduser()
        if requested_path.exists():
            local_model_path = str(requested_path.resolve())
        else:
            try:
                local_model_path = snapshot_download(
                    pretrained_model_name_or_path,
                    repo_type="model",
                )
            except (HFValidationError, RepositoryNotFoundError) as error:
                raise FileNotFoundError(
                    f"Base model was not found locally or on Hugging Face: "
                    f"{pretrained_model_name_or_path}"
                ) from error

        if use_latent_motion_queries:
            model_config = kwargs.pop("config", None)
            if model_config is None:
                model_config = cls.config_class.from_pretrained(local_model_path)
            elif isinstance(model_config, (str, Path)):
                model_config = cls.config_class.from_pretrained(model_config)
            elif isinstance(model_config, Mapping):
                model_config = cls.config_class.from_dict(
                    _to_plain_config(model_config)
                )

            action_head_cfg = dict(model_config.action_head_cfg)
            action_head_cfg.update(
                {
                    "use_latent_motion_queries": True,
                    "tune_latent_motion_tokenizer": tune_latent_motion_tokenizer,
                    "latent_motion_token_count": latent_motion_token_count,
                    "latent_motion_codebook_dim": latent_motion_codebook_dim,
                    "latent_motion_hidden_layer": latent_motion_hidden_layer,
                    "latent_loss_weight": latent_loss_weight,
                    "tokenizer_vae_loss_weight": tokenizer_vae_loss_weight,
                }
            )
            model_config.action_head_cfg = action_head_cfg
            kwargs["config"] = model_config

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )

        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )

        if use_latent_motion_queries:
            if not latent_motion_tokenizer_path:
                raise ValueError(
                    "latent_motion_tokenizer_path is required when "
                    "use_latent_motion_queries=true"
                )
            pretrained_model.action_head.configure_latent_motion(
                tokenizer_path=latent_motion_tokenizer_path,
                image_encoder_path=latent_motion_image_encoder_path,
                tune_tokenizer=tune_latent_motion_tokenizer,
                token_count=latent_motion_token_count,
                codebook_dim=latent_motion_codebook_dim,
                hidden_layer=latent_motion_hidden_layer,
                latent_loss_weight=latent_loss_weight,
                tokenizer_vae_loss_weight=tokenizer_vae_loss_weight,
            )
            action_head_cfg = dict(pretrained_model.config.action_head_cfg)
            action_head_cfg.update(
                {
                    "use_latent_motion_queries": True,
                    "tune_latent_motion_tokenizer": tune_latent_motion_tokenizer,
                    "latent_motion_token_count": latent_motion_token_count,
                    "latent_motion_codebook_dim": latent_motion_codebook_dim,
                    "latent_motion_hidden_layer": latent_motion_hidden_layer,
                    "latent_loss_weight": latent_loss_weight,
                    "tokenizer_vae_loss_weight": tokenizer_vae_loss_weight,
                }
            )
            pretrained_model.config.action_head_cfg = action_head_cfg

        if reinitialize_action_head:
            for module in (
                pretrained_model.action_head.state_encoder,
                pretrained_model.action_head.action_encoder,
                pretrained_model.action_head.action_decoder,
                pretrained_model.action_head.vlln,
                pretrained_model.action_head.vl_self_attention,
                pretrained_model.action_head.model,
            ):
                reset_parameters(module)
            if hasattr(pretrained_model.action_head, "position_embedding"):
                torch.nn.init.normal_(
                    pretrained_model.action_head.position_embedding.weight,
                    mean=0.0,
                    std=0.02,
                )
            if pretrained_model.action_head.pred_latent_motion_head is not None:
                pretrained_model.action_head.pred_latent_motion_head.reset_parameters()

        return pretrained_model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
