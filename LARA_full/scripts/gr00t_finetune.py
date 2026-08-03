"""Train or downstream-post-train LARA with a latent-motion tokenizer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lara_full.experiment.runner import TrainRunner
from lara_full.model.gr00t_n1 import GR00T_N1_5
from lara_full.utils.peft import get_lora_model
from lara_full.utils.train import dataset_factory


def main(config: DictConfig):
    train_dataset = dataset_factory(config)

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
        use_latent_motion_queries=config.use_latent_motion_queries,
        latent_motion_tokenizer_path=config.latent_motion_tokenizer_path,
        latent_motion_image_encoder_path=config.latent_motion_image_encoder_path,
        tune_latent_motion_tokenizer=config.tune_latent_motion_tokenizer,
        latent_motion_token_count=config.latent_motion_token_count,
        latent_motion_codebook_dim=config.latent_motion_codebook_dim,
        latent_motion_hidden_layer=config.latent_motion_hidden_layer,
        latent_loss_weight=config.latent_loss_weight,
        tokenizer_vae_loss_weight=config.tokenizer_vae_loss_weight,
        reinitialize_action_head=config.reinitialize_action_head,
        ignore_mismatched_sizes=True,
    )
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.get("run_name"),
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=config.report_to,
        seed=config.seed,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
    )

    TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    ).train()


@hydra.main(version_base=None, config_path="../lara_full/config", config_name="mani_test_all")
def hydra_main(config: DictConfig):
    print(OmegaConf.to_yaml(config, resolve=True))

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if config.num_gpus < 1 or config.num_gpus > available_gpus:
        raise ValueError(
            f"Requested {config.num_gpus} GPUs, but {available_gpus} are visible"
        )

    if config.num_gpus == 1 or os.environ.get("IS_TORCHRUN") == "1":
        main(config)
        return

    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={config.num_gpus}",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    environment = os.environ.copy()
    environment["IS_TORCHRUN"] = "1"
    raise SystemExit(subprocess.run(command, env=environment, check=False).returncode)


if __name__ == "__main__":
    hydra_main()
