"""Latent motion tokenizer model."""

from dataclasses import dataclass
from typing import Any

import hydra
import lpips
import omegaconf
import torch
from torch import nn
import torch.nn.functional as functional
from transformers import PreTrainedModel
from transformers import ViTMAEModel
from transformers.modeling_outputs import ModelOutput

from moto.latent_motion_tokenizer.configs import LatentMotionTokenizerConfig


@dataclass
class LatentMotionTokenizerOutput(ModelOutput):
    """Outputs returned by the latent motion tokenizer.

    `embed` contains the latent-motion tokens used by LARA representation
    alignment, while `quant` contains the codebook-quantized tokens.
    """

    loss: torch.FloatTensor | None = None
    commit_loss: torch.FloatTensor | None = None
    recons_loss: torch.FloatTensor | None = None
    recons_hidden_loss: torch.FloatTensor | None = None
    perceptual_loss: torch.FloatTensor | None = None
    active_code_num: torch.FloatTensor | None = None
    recons_pixel_values: torch.FloatTensor | None = None
    indices: torch.LongTensor | None = None
    embed: torch.FloatTensor | None = None
    quant: torch.FloatTensor | None = None


class LatentMotionTokenizer(PreTrainedModel):
    config_class = LatentMotionTokenizerConfig

    def __init__(self, config: LatentMotionTokenizerConfig):
        super().__init__(config)

        self.codebook_dim = config.codebook_dim
        image_encoder_config = omegaconf.DictConfig(config.image_encoder_config)
        m_former_config = omegaconf.DictConfig(config.m_former_config)
        vector_quantizer_config = omegaconf.DictConfig(config.vector_quantizer_config)
        decoder_config = omegaconf.DictConfig(config.decoder_config)
        hidden_state_decoder_config = (
            omegaconf.DictConfig(config.hidden_state_decoder_config)
            if config.hidden_state_decoder_config is not None
            else None
        )

        self.image_encoder = hydra.utils.instantiate(image_encoder_config).requires_grad_(False).eval()
        self.m_former = hydra.utils.instantiate(m_former_config)
        self.vector_quantizer = hydra.utils.instantiate(vector_quantizer_config)
        self.decoder = hydra.utils.instantiate(decoder_config)
        self.hidden_state_decoder = (
            hydra.utils.instantiate(hidden_state_decoder_config) if hidden_state_decoder_config is not None else None
        )

        decoder_hidden_size = decoder_config.config.hidden_size
        m_former_hidden_size = m_former_config.config.hidden_size

        if isinstance(self.image_encoder, ViTMAEModel):
            self.image_encoder.config.mask_ratio = 0.0

        self.vq_down_resampler = nn.Sequential(
            nn.Linear(m_former_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, self.codebook_dim),
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(self.codebook_dim, self.codebook_dim),
            nn.Tanh(),
            nn.Linear(self.codebook_dim, decoder_hidden_size),
        )
        self.loss_fn_lpips = None
        if config.perceptual_loss_w > 0:
            self.loss_fn_lpips = lpips.LPIPS(net="vgg", verbose=False).requires_grad_(False).eval()

    def get_state_dict_to_save(self) -> dict[str, Any]:
        """Return trainable tokenizer state without the frozen external encoders."""

        excluded_modules = ("loss_fn_lpips", "image_encoder")
        return {
            key: value
            for key, value in self.state_dict().items()
            if not any(module_name in key for module_name in excluded_modules)
        }

    @torch.no_grad()
    def decode_image(
        self,
        cond_pixel_values: torch.Tensor,
        given_motion_token_ids: torch.Tensor,
    ) -> LatentMotionTokenizerOutput:
        """Decode target images from condition images and codebook indices."""

        quant = self.vector_quantizer.get_codebook_entry(given_motion_token_ids)
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(
            cond_input=cond_pixel_values,
            latent_motion_tokens=latent_motion_tokens_up,
        )
        return LatentMotionTokenizerOutput(recons_pixel_values=recons_pixel_values)

    def forward(
        self,
        cond_pixel_values: torch.Tensor,
        target_pixel_values: torch.Tensor,
        return_recons_only: bool = False,
        return_motion_token_ids_only: bool = False,
    ) -> LatentMotionTokenizerOutput:
        """Encode motion, quantize it, and reconstruct the target image."""

        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state

        query_num = self.m_former.query_num
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states,
        ).last_hidden_state[:, :query_num]

        latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
        quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)

        if return_motion_token_ids_only:
            return LatentMotionTokenizerOutput(indices=indices)

        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(
            cond_input=cond_pixel_values,
            latent_motion_tokens=latent_motion_tokens_up,
        )

        if return_recons_only:
            return LatentMotionTokenizerOutput(
                recons_pixel_values=recons_pixel_values,
                indices=indices,
            )

        if self.config.use_abs_recons_loss:
            recons_loss = torch.abs(recons_pixel_values - target_pixel_values).mean()
        else:
            recons_loss = functional.mse_loss(target_pixel_values, recons_pixel_values)

        if self.config.perceptual_loss_w > 0:
            if self.loss_fn_lpips is None:
                raise RuntimeError("LPIPS is required when perceptual_loss_w is positive.")
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips(
                    target_pixel_values,
                    recons_pixel_values,
                    normalize=True,
                ).mean()
        else:
            perceptual_loss = torch.zeros_like(recons_loss)

        loss = (
            self.config.commit_loss_w * commit_loss
            + self.config.recon_loss_w * recons_loss
            + self.config.perceptual_loss_w * perceptual_loss
        )

        recons_hidden_loss = None
        if self.hidden_state_decoder is not None:
            recons_hidden_states = self.hidden_state_decoder(
                cond_input=cond_hidden_states,
                latent_motion_tokens=latent_motion_tokens_up,
            )
            recons_hidden_loss = functional.mse_loss(target_hidden_states, recons_hidden_states)
            loss += self.config.recon_hidden_loss_w * recons_hidden_loss

        active_code_num = indices.unique().numel()
        active_code_num = loss.new_tensor(active_code_num)

        return LatentMotionTokenizerOutput(
            loss=loss,
            commit_loss=commit_loss,
            recons_loss=recons_loss,
            recons_hidden_loss=recons_hidden_loss,
            perceptual_loss=perceptual_loss,
            active_code_num=active_code_num,
            indices=indices,
            embed=latent_motion_tokens_down,
            quant=quant,
        )
