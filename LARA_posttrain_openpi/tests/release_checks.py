"""Dependency-free checks for the release tree."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_LAUNCHERS = {"run_pi05_lara_libero.sh"}
EXPECTED_MOTO_FILES = {
    "moto/__init__.py",
    "moto/latent_motion_tokenizer/__init__.py",
    "moto/latent_motion_tokenizer/configs.py",
    "moto/latent_motion_tokenizer/loading.py",
    "moto/latent_motion_tokenizer/model/__init__.py",
    "moto/latent_motion_tokenizer/model/latent_motion_decoder.py",
    "moto/latent_motion_tokenizer/model/latent_motion_tokenizer.py",
    "moto/latent_motion_tokenizer/model/m_former.py",
    "moto/latent_motion_tokenizer/model/modeling_lmd_vit.py",
    "moto/latent_motion_tokenizer/model/vector_quantizer.py",
}
FORBIDDEN_DIRECTORIES = {
    ".venv",
    "data",
    "moto_gpt_pretrained_model",
    "qsub_output",
    "wandb",
    "moto/configs",
    "moto/experiment",
    "moto/hydra_outputs",
    "moto/moto_gpt",
}
FORBIDDEN_TEXT = {
    "/home/lmy",
    "/share/generalvision",
    "/mnt/data0",
    "/mnt/data3",
    "HumanoidVLA",
    "Video_Depth_Anything",
    "VideoFlow",
    "flow_viz",
    "lapa_tokenizer",
    "breakpoint(",
    "pdb.set_trace",
    "TORCH_DISTRIBUTED_DEBUG",
    "git checkout",
}
FORBIDDEN_PYTHON_TEXT = {
    "depth_image",
    '"_depth"',
    "is_debug",
}
SCANNED_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
EXCLUDED_TOP_LEVEL = {".git", "third_party"}


def _source_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.relative_to(ROOT).parts[0] not in EXCLUDED_TOP_LEVEL
        and "__pycache__" not in path.parts
        and (path.suffix in SCANNED_SUFFIXES or path.name == ".env.example")
    ]


def main() -> int:
    errors: list[str] = []

    launchers = {path.name for path in ROOT.glob("run*.sh")}
    if launchers != EXPECTED_LAUNCHERS:
        errors.append(f"Unexpected root launchers: {sorted(launchers)}")
    launcher_path = ROOT / "run_pi05_lara_libero.sh"
    if launcher_path.is_file() and not launcher_path.stat().st_mode & 0o111:
        errors.append("The LIBERO launcher is not executable")

    moto_files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "moto").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if moto_files != EXPECTED_MOTO_FILES:
        errors.append(
            "Moto release files differ from the allowlist: "
            f"missing={sorted(EXPECTED_MOTO_FILES - moto_files)}, "
            f"extra={sorted(moto_files - EXPECTED_MOTO_FILES)}"
        )
    errors.extend(
        f"Moto library file should not be executable: {path.relative_to(ROOT)}"
        for path in (ROOT / "moto").rglob("*.py")
        if path.stat().st_mode & 0o111
    )

    errors.extend(
        f"Forbidden release directory exists: {relative_path}"
        for relative_path in sorted(FORBIDDEN_DIRECTORIES)
        if (ROOT / relative_path).exists()
    )

    source_files = _source_files()
    for path in source_files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if path.resolve() != Path(__file__).resolve():
            errors.extend(
                f"Forbidden text {pattern!r} in {path.relative_to(ROOT)}"
                for pattern in FORBIDDEN_TEXT
                if pattern in text
            )

        if path.suffix == ".py":
            if b"\r\n" in path.read_bytes():
                errors.append(f"CRLF line endings in {path.relative_to(ROOT)}")
            if path.resolve() != Path(__file__).resolve():
                errors.extend(
                    f"Forbidden Python text {pattern!r} in {path.relative_to(ROOT)}"
                    for pattern in FORBIDDEN_PYTHON_TEXT
                    if pattern in text
                )
                if any(line.lstrip().startswith("###") for line in text.splitlines()):
                    errors.append(f"Python heading with three or more hashes in {path.relative_to(ROOT)}")
            try:
                ast.parse(text, filename=str(path))
            except SyntaxError as exc:
                errors.append(f"Python syntax error in {path.relative_to(ROOT)}: {exc}")

    pi0_source = (ROOT / "src/openpi/models_pytorch/pi0_pytorch.py").read_text(encoding="utf-8")
    if "target_motion_embed = outputs.embed.reshape" not in pi0_source:
        errors.append("LARA supervision must read the tokenizer's embed output")
    if "outputs.quant" in pi0_source:
        errors.append("LARA supervision must not use the tokenizer's quantized output")

    training_config_source = (ROOT / "src/openpi/training/config.py").read_text(encoding="utf-8")
    if "latent_motion_loss_weight=0.01" not in training_config_source:
        errors.append("pi05_lara_libero must use latent_motion_loss_weight=0.01")
    if "tokenizer_loss_weight=0.01" not in training_config_source:
        errors.append("pi05_lara_libero must use tokenizer_loss_weight=0.01")

    training_source = (ROOT / "scripts/train_pytorch.py").read_text(encoding="utf-8")
    if "Use --resume or --overwrite to indicate how to handle it." not in training_source:
        errors.append("PyTorch training must refuse to overwrite an existing run implicitly")
    if "_prune_checkpoints(config.checkpoint_dir, global_step, config.keep_period)" not in training_source:
        errors.append("PyTorch checkpoint retention must honor keep_period")

    tokenizer_source = (ROOT / "moto/latent_motion_tokenizer/model/latent_motion_tokenizer.py").read_text(
        encoding="utf-8"
    )
    lpips_no_grad = "with torch.no_grad():\n                perceptual_loss = self.loss_fn_lpips("
    if lpips_no_grad not in tokenizer_source:
        errors.append("LPIPS forward must remain under no_grad to match the released Stage 3 experiment")

    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1

    print("Release checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
