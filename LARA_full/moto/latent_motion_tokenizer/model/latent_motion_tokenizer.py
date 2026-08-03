"""Latent Motion Tokenizer model."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import hydra
import lpips
import omegaconf
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, ViTMAEModel
from transformers.modeling_outputs import ModelOutput

from moto.latent_motion_tokenizer.configs import LatentMotionTokenizerConfig


@dataclass
class LatentMotionTokenizerOutput(ModelOutput):
    """
    Output type of LatentMotionTokenizer.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Total loss (if training).
        commit_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Vector quantization commitment loss.
        recons_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Reconstruction loss.
        recons_hidden_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Hidden state reconstruction loss.
        perceptual_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Perceptual loss.
        active_code_num (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Number of unique codebook entries used.
        recons_pixel_values (`torch.FloatTensor`, *optional*):
            Reconstructed pixel values.
        indices (`torch.LongTensor`, *optional*):
            Indices of selected codebook entries.
        embed (`torch.FloatTensor`, *optional*):
            Latent-motion tokens used by LARA representation alignment.
        quant (`torch.FloatTensor`, *optional*):
            Codebook-quantized latent-motion tokens.
    """
    loss: Optional[torch.FloatTensor] = None
    commit_loss: Optional[torch.FloatTensor] = None
    recons_loss: Optional[torch.FloatTensor] = None
    recons_hidden_loss: Optional[torch.FloatTensor] = None
    perceptual_loss: Optional[torch.FloatTensor] = None
    active_code_num: Optional[torch.FloatTensor] = None
    recons_pixel_values: Optional[torch.FloatTensor] = None
    indices: Optional[torch.LongTensor] = None
    embed: Optional[torch.FloatTensor] = None
    quant: Optional[torch.FloatTensor] = None


class LatentMotionTokenizer(PreTrainedModel):
    config_class = LatentMotionTokenizerConfig

    def __init__(
        self,
        config: LatentMotionTokenizerConfig,
    ):
        super().__init__(config)

        self.codebook_dim = config.codebook_dim
        image_encoder_config = omegaconf.DictConfig(config.image_encoder_config)
        m_former_config = omegaconf.DictConfig(config.m_former_config)
        vector_quantizer_config = omegaconf.DictConfig(config.vector_quantizer_config)
        decoder_config = omegaconf.DictConfig(config.decoder_config)
        hidden_state_decoder_config = omegaconf.DictConfig(config.hidden_state_decoder_config) if config.hidden_state_decoder_config is not None else None

        self.image_encoder = hydra.utils.instantiate(image_encoder_config).requires_grad_(False).eval()
        self.m_former = hydra.utils.instantiate(m_former_config)
        self.vector_quantizer = hydra.utils.instantiate(vector_quantizer_config)
        self.decoder = hydra.utils.instantiate(decoder_config)
        self.hidden_state_decoder = hydra.utils.instantiate(hidden_state_decoder_config) if hidden_state_decoder_config is not None else None

        decoder_hidden_size = decoder_config.config.hidden_size # type: ignore
        m_former_hidden_size = m_former_config.config.hidden_size # type: ignore

        if isinstance(self.image_encoder, ViTMAEModel):
            self.image_encoder.config.mask_ratio = 0.0

        self.vq_down_resampler = nn.Sequential(
            nn.Linear(m_former_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, self.codebook_dim)
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(self.codebook_dim, self.codebook_dim),
            nn.Tanh(),
            nn.Linear(self.codebook_dim, decoder_hidden_size)
        )

        self.loss_fn_lpips = lpips.LPIPS(net="vgg").requires_grad_(False).eval()

    def get_state_dict_to_save(self) -> Dict[str, Any]:
        """Get state dict excluding certain modules."""
        modules_to_exclude = ['loss_fn_lpips', 'image_encoder']
        return {k: v for k, v in self.state_dict().items() if
                not any(module_name in k for module_name in modules_to_exclude)}

    @torch.no_grad()
    def decode_image(
        self,
        cond_pixel_values: torch.Tensor,
        given_motion_token_ids: torch.Tensor
    ) -> LatentMotionTokenizerOutput:
        """Decode images from condition and motion tokens."""
        quant = self.vector_quantizer.get_codebook_entry(given_motion_token_ids)
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(
            cond_input=cond_pixel_values,
            latent_motion_tokens=latent_motion_tokens_up
        )
        return LatentMotionTokenizerOutput(recons_pixel_values=recons_pixel_values)

    def forward(
        self,
        cond_pixel_values: torch.Tensor,
        target_pixel_values: torch.Tensor,
        return_recons_only: bool = False,
        return_motion_token_ids_only: bool = False,
    ) -> LatentMotionTokenizerOutput:
        """
        Forward pass of the model.

        Args:
            cond_pixel_values (`torch.Tensor`):
                Conditional input images.
            target_pixel_values (`torch.Tensor`):
                Target images to reconstruct.
            return_recons_only (`bool`, *optional*, defaults to False):
                Whether to return only reconstructed images.
            return_motion_token_ids_only (`bool`, *optional*, defaults to False):
                Whether to return only motion token indices.

        Returns:
            [`LatentMotionTokenizerOutput`]: Model outputs.
        """
        # Tokenization
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

        # Detokenization
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(
            cond_input=cond_pixel_values,
            latent_motion_tokens=latent_motion_tokens_up
        )

        if return_recons_only:
            return LatentMotionTokenizerOutput(
                recons_pixel_values=recons_pixel_values,
                indices=indices
            )

        # Compute losses
        # if self.config.use_abs_recons_loss:
        #     recons_loss = torch.abs(recons_pixel_values - target_pixel_values).mean()
        # else:
        recons_loss = F.mse_loss(target_pixel_values, recons_pixel_values)

        if self.config.perceptual_loss_w > 0:
            perceptual_loss = self.loss_fn_lpips(
                target_pixel_values, recons_pixel_values, normalize=True
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
                latent_motion_tokens=latent_motion_tokens_up
            )
            recons_hidden_loss = F.mse_loss(target_hidden_states, recons_hidden_states)
            loss += self.config.recon_hidden_loss_w * recons_hidden_loss

        active_code_num = torch.tensor(torch.unique(indices).shape[0]).float().to(loss.device)

        return LatentMotionTokenizerOutput(
            loss=loss,
            active_code_num=active_code_num,
            indices=indices,
            embed=latent_motion_tokens_down,
            quant=quant,
        )
