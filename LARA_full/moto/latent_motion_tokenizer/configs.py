from transformers import PretrainedConfig
from omegaconf import DictConfig, OmegaConf
from typing import Optional

class LatentMotionTokenizerConfig(PretrainedConfig):
    """Configuration class for LatentMotionTokenizer.

    Args:
        codebook_dim (`int`, *optional*, defaults to 32):
            Dimension of the codebook embeddings.
        commit_loss_w (`float`, *optional*, defaults to 1.0):
            Weight for commitment loss.
        recon_loss_w (`float`, *optional*, defaults to 1.0):
            Weight for reconstruction loss.
        recon_hidden_loss_w (`float`, *optional*, defaults to 1.0):
            Weight for hidden state reconstruction loss.
        perceptual_loss_w (`float`, *optional*, defaults to 1.0):
            Weight for perceptual loss.
        use_abs_recons_loss (`bool`, *optional*, defaults to False):
            Whether to use absolute reconstruction loss.
    """
    model_type = "latent_motion_tokenizer"

    def __init__(
        self,
        codebook_dim=32,
        commit_loss_w=1.0,
        recon_loss_w=1.0,
        recon_hidden_loss_w=1.0,
        perceptual_loss_w=1.0,
        use_abs_recons_loss=False,

        image_encoder_config: DictConfig = None,
        m_former_config: DictConfig = None,
        vector_quantizer_config: DictConfig = None,
        decoder_config: DictConfig = None,
        hidden_state_decoder_config: Optional[DictConfig | None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_dim = codebook_dim
        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.recon_hidden_loss_w = recon_hidden_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.use_abs_recons_loss = use_abs_recons_loss

        self.image_encoder_config = OmegaConf.to_container(image_encoder_config) if isinstance(image_encoder_config, DictConfig) else image_encoder_config
        self.m_former_config = OmegaConf.to_container(m_former_config) if isinstance(m_former_config, DictConfig) else m_former_config
        self.vector_quantizer_config = OmegaConf.to_container(vector_quantizer_config) if isinstance(vector_quantizer_config, DictConfig) else vector_quantizer_config
        self.decoder_config = OmegaConf.to_container(decoder_config) if isinstance(decoder_config, DictConfig) else decoder_config
        self.hidden_state_decoder_config = OmegaConf.to_container(hidden_state_decoder_config) if isinstance(hidden_state_decoder_config, DictConfig) else hidden_state_decoder_config
