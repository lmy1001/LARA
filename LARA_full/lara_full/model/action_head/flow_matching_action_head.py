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

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from lara_full.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from moto.latent_motion_tokenizer.loading import load_latent_motion_tokenizer

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(torch.empty(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.empty(num_categories, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W, mean=0.0, std=0.02)
        nn.init.zeros_(self.b)

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=64, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    use_latent_motion_queries: bool = field(default=False)
    latent_motion_tokenizer_path: Optional[str] = field(default=None)
    latent_motion_image_encoder_path: Optional[str] = field(default=None)
    tune_latent_motion_tokenizer: bool = field(default=False)
    latent_motion_codebook_dim: int = field(default=32)
    latent_motion_token_count: int = field(default=8)
    latent_motion_hidden_layer: int = field(default=-3)
    latent_loss_weight: float = field(default=0.01)
    tokenizer_vae_loss_weight: float = field(default=0.01)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        self.use_latent_motion_queries = bool(
            getattr(config, "use_latent_motion_queries", False)
        )
        self.latent_motion_tokenizer = None
        self.tune_latent_motion_tokenizer = bool(
            getattr(config, "tune_latent_motion_tokenizer", False)
        )
        self.latent_motion_codebook_dim = int(
            getattr(config, "latent_motion_codebook_dim", 32)
        )
        self.latent_motion_token_count = int(
            getattr(config, "latent_motion_token_count", 8)
        )
        self.latent_motion_hidden_layer = int(
            getattr(config, "latent_motion_hidden_layer", -3)
        )
        self.latent_loss_weight = float(getattr(config, "latent_loss_weight", 0.01))
        self.tokenizer_vae_loss_weight = float(
            getattr(config, "tokenizer_vae_loss_weight", 0.01)
        )
        self.pred_latent_motion_head = None
        if self.use_latent_motion_queries:
            self._initialize_latent_motion_head()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def _initialize_latent_motion_head(self):
        output_dim = self.latent_motion_token_count * self.latent_motion_codebook_dim
        if (
            self.pred_latent_motion_head is None
            or self.pred_latent_motion_head.out_features != output_dim
        ):
            self.pred_latent_motion_head = nn.Linear(self.input_embedding_dim, output_dim)

    def configure_latent_motion(
        self,
        *,
        tokenizer_path: str,
        image_encoder_path: str | None = None,
        tune_tokenizer: bool = False,
        token_count: int = 8,
        codebook_dim: int = 32,
        hidden_layer: int = -3,
        latent_loss_weight: float = 0.01,
        tokenizer_vae_loss_weight: float = 0.01,
    ):
        """Attach the auxiliary tokenizer once, outside model construction."""

        self.use_latent_motion_queries = True
        self.tune_latent_motion_tokenizer = tune_tokenizer
        self.latent_motion_token_count = token_count
        self.latent_motion_codebook_dim = codebook_dim
        self.latent_motion_hidden_layer = hidden_layer
        self.latent_loss_weight = latent_loss_weight
        self.tokenizer_vae_loss_weight = tokenizer_vae_loss_weight
        self._initialize_latent_motion_head()

        self.latent_motion_tokenizer = load_latent_motion_tokenizer(
            tokenizer_path,
            image_encoder_path=image_encoder_path,
        )
        self.latent_motion_tokenizer.requires_grad_(tune_tokenizer)
        self.latent_motion_tokenizer.image_encoder.requires_grad_(False)
        self.latent_motion_tokenizer.loss_fn_lpips.requires_grad_(False)

        for key, value in {
            "use_latent_motion_queries": True,
            "tune_latent_motion_tokenizer": tune_tokenizer,
            "latent_motion_codebook_dim": codebook_dim,
            "latent_motion_token_count": token_count,
            "latent_motion_hidden_layer": hidden_layer,
            "latent_loss_weight": latent_loss_weight,
            "tokenizer_vae_loss_weight": tokenizer_vae_loss_weight,
        }.items():
            setattr(self.config, key, value)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

        # if self.freeze_decode_layer:
        #     self.decode_layer.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if self.use_latent_motion_queries and self.latent_motion_tokenizer is not None:
                if not self.tune_latent_motion_tokenizer:
                    self.latent_motion_tokenizer.eval()
                self.latent_motion_tokenizer.image_encoder.eval()
                self.latent_motion_tokenizer.loss_fn_lpips.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature, inputs) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state[:, :1], embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        vl_embs = vl_embeds
        vl_attn_mask = backbone_output.backbone_attention_mask

        model_outputs = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=self.use_latent_motion_queries,
        )
        if self.use_latent_motion_queries:
            model_output, all_hidden_states = model_outputs
        else:
            model_output = model_outputs
            all_hidden_states = None

        action_start = state_features.shape[1]
        action_end = state_features.shape[1] + action_features.shape[1]
        pred = self.action_decoder(model_output[:, action_start:action_end], embodiment_id)
        pred_actions = pred[:, -actions.shape[1]:]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()

        output_dict = {
            "loss": loss,
        }

        if self.use_latent_motion_queries:
            if self.latent_motion_tokenizer is None:
                raise RuntimeError(
                    "Latent-motion supervision is enabled, but no tokenizer was attached. "
                    "Set latent_motion_tokenizer_path in the training config."
                )
            images = inputs["images"].squeeze(1)
            if images.ndim != 5 or images.shape[1] < 2:
                raise ValueError(
                    "Latent-motion supervision requires at least two video frames per sample"
                )

            batch_size, frame_count, channels, height, width = images.shape
            tokenizer_outputs = self.latent_motion_tokenizer(
                cond_pixel_values=images[:, :1]
                .repeat(1, frame_count - 1, 1, 1, 1)
                .reshape(-1, channels, height, width),
                target_pixel_values=images[:, 1:].reshape(
                    -1, channels, height, width
                ),
            )
            tokenizer_vae_loss = tokenizer_outputs.loss.mean()
            if tokenizer_outputs.embed is None:
                raise RuntimeError(
                    "Latent-motion tokenizer did not return its embed output"
                )
            target_motion_embed = tokenizer_outputs.embed.reshape(
                batch_size, frame_count - 1, -1
            ).mean(dim=1)

            if all_hidden_states is None:
                raise RuntimeError("DiT hidden states were not returned")
            hidden_state = all_hidden_states[self.latent_motion_hidden_layer][:, -1]
            predicted_motion_embed = self.pred_latent_motion_head(hidden_state)
            latent_loss = (
                1
                - F.cosine_similarity(
                    predicted_motion_embed,
                    target_motion_embed.to(predicted_motion_embed.dtype),
                    dim=-1,
                )
            ).mean()

            output_dict.update(
                {
                    "latent_loss": latent_loss,
                    "tokenizer_vae_loss": tokenizer_vae_loss,
                    "unique_code_count": tokenizer_outputs.active_code_num,
                }
            )
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state[:, :1], embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            vl_embs = vl_embeds

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)
            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )

            action_start = state_features.shape[1]
            action_end = state_features.shape[1] + action_features.shape[1]
            pred = self.action_decoder(model_output[:, action_start:action_end], embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity

        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
