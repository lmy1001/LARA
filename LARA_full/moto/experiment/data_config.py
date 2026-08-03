"""Data configuration for latent-motion-tokenizer pretraining."""

from lara_full.data.dataset import ModalityConfig
from lara_full.data.transform.base import ComposedModalityTransform, ModalityTransform
from lara_full.data.transform.concat import ConcatTransform
from lara_full.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from lara_full.experiment.data_config import BaseDataConfig


class MotoTokenizerDataConfig(BaseDataConfig):
    """Produce the two-frame video samples consumed by the tokenizer collator."""

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            )
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
                ConcatTransform(video_concat_order=self.video_keys),
            ]
        )
