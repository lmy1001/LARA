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


import os
from typing import Optional

import torch
import transformers
from safetensors.torch import save_file
from torch.utils.data import Dataset, Sampler
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class DualBrainTrainer(transformers.Trainer):
    def __init__(self, **kwargs):
        self.compute_dtype = kwargs.pop("compute_dtype")
        self.extra_args = kwargs.pop("extra_args", None)
        super().__init__(**kwargs)

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]
        total_loss = loss
        metrics = {"action_loss": loss.detach()}

        if "latent_loss" in outputs:
            latent_loss = outputs["latent_loss"]
            tokenizer_vae_loss = outputs["tokenizer_vae_loss"]
            unwrapped_model = self.accelerator.unwrap_model(model)
            action_head = unwrapped_model.action_head
            total_loss = (
                total_loss
                + action_head.latent_loss_weight * latent_loss
                + action_head.tokenizer_vae_loss_weight * tokenizer_vae_loss
            )
            metrics.update(
                {
                    "latent_loss": latent_loss.detach(),
                    "tokenizer_vae_loss": tokenizer_vae_loss.detach(),
                    "unique_code_count": outputs["unique_code_count"].detach(),
                }
            )

        metrics["total_loss"] = total_loss.detach()
        logging_steps = max(int(self.args.logging_steps), 1)
        if self.state.global_step % logging_steps == 0:
            self.log({key: value.float().item() for key, value in metrics.items()})
        return (total_loss, outputs) if return_outputs else total_loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False,
    ):
        """Save the trainable model and a portable tokenizer checkpoint."""
        output_dir = output_dir or self.args.output_dir
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.args.should_save and self.accelerator.is_main_process:
            tokenizer_prefixes = (
                "action_head.latent_motion_tokenizer.image_encoder.",
                "action_head.latent_motion_tokenizer.loss_fn_lpips.",
            )
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith(tokenizer_prefixes)
            }

            unwrapped_model = self.accelerator.unwrap_model(self.model)
            tokenizer = unwrapped_model.action_head.latent_motion_tokenizer
            if tokenizer is not None:
                tokenizer_dir = os.path.join(output_dir, "latent_motion_tokenizer")
                os.makedirs(tokenizer_dir, exist_ok=True)
                tokenizer.config.save_pretrained(tokenizer_dir)
                tokenizer_exclusions = ("image_encoder.", "loss_fn_lpips.")
                tokenizer_state = {
                    key: value
                    for key, value in tokenizer.state_dict().items()
                    if not key.startswith(tokenizer_exclusions)
                }
                save_file(
                    tokenizer_state,
                    os.path.join(tokenizer_dir, "model.safetensors"),
                )

            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        output = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        return output
