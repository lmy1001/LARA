"""Training runner for the released latent-motion tokenizer."""

import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, set_seed

from lara_full.experiment.runner import BaseRunner
from lara_full.utils.experiment import CheckpointFormatCallback
from moto.experiment.trainer import MotoTokenizerTrainer
from moto.latent_motion_tokenizer.model.latent_motion_tokenizer import (
    LatentMotionTokenizer,
)
from moto.utils.collators import DefaultDataCollatorMotoTokenizer


class MotoRunner(BaseRunner):
    """Configure Hugging Face Trainer for tokenizer pretraining."""

    def __init__(
        self,
        model: LatentMotionTokenizer,
        training_args: TrainingArguments,
        train_dataset: Dataset,
        resume_from_checkpoint: bool = False,
    ):
        super().__init__(
            model=model,
            training_args=training_args,
            extra_args=None,
            train_dataset=train_dataset,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        training_args.run_name = training_args.run_name or Path(
            training_args.output_dir
        ).name
        print(f"Run name: {training_args.run_name}")

        compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
        set_seed(training_args.seed)
        self.trainer = self.create_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            data_collator=DefaultDataCollatorMotoTokenizer(),
            compute_dtype=compute_dtype,
        )

        self.rank = int(os.environ.get("RANK", 0))
        if self.rank == 0:
            metadata_json = {}
            metadata_path = self.exp_cfg_dir / "metadata.json"
            if metadata_path.exists():
                with metadata_path.open(encoding="utf-8") as handle:
                    metadata_json = json.load(handle)

            if hasattr(train_dataset, "tag") and hasattr(train_dataset, "metadata"):
                metadata_json[train_dataset.tag] = train_dataset.metadata.model_dump(
                    mode="json"
                )
            elif hasattr(train_dataset, "merged_metadata"):
                metadata_json.update(
                    {
                        tag: metadata.model_dump(mode="json")
                        for tag, metadata in train_dataset.merged_metadata.items()
                    }
                )
            else:
                raise TypeError(
                    f"Unsupported training dataset type: {type(train_dataset).__name__}"
                )

            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata_json, handle, indent=4)

        if training_args.report_to == "wandb":
            os.environ.setdefault("WANDB_PROJECT", "lara-tokenizer")
            runtime_id = os.environ.get("RUNTIME_ID")
            if runtime_id:
                os.environ.setdefault("WANDB_RUN_ID", runtime_id)
            os.environ["WANDB_DIR"] = training_args.output_dir
            with (self.output_dir / "wandb_config.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "project": os.environ.get("WANDB_PROJECT", ""),
                        "run_id": os.environ.get("WANDB_RUN_ID", ""),
                    },
                    handle,
                )
            training_args.report_to = ["wandb"]
        else:
            tensorboard_dir = self.output_dir / "runs"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
            training_args.report_to = ["tensorboard"]

    def create_trainer(
        self,
        model,
        training_args,
        train_dataset,
        data_collator,
        compute_dtype,
        global_batch_size=None,
    ):
        if global_batch_size is not None:
            device_count = max(torch.cuda.device_count(), 1)
            local_batch_size = training_args.per_device_train_batch_size
            training_args.gradient_accumulation_steps = max(
                1,
                global_batch_size // (local_batch_size * device_count),
            )

        trainer = MotoTokenizerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )
        trainer.add_callback(
            CheckpointFormatCallback(
                run_name=training_args.run_name,
                exp_cfg_dir=self.exp_cfg_dir,
            )
        )

        print(
            f"train dataloader length: {len(trainer.get_train_dataloader())}\n"
            f"train dataset length: {len(trainer.train_dataset)}\n"
            f"GPU memory before training: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            flush=True,
        )
        return trainer
