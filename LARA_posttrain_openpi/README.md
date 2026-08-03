<div align="center">

<h1>LARA: Latent Action Representation Alignment for Vision-Language-Action Models</h1>

<p><strong>ICML 2026</strong></p>

<p>
Mengya Liu<sup>*1</sup>, Baoxiong Jia<sup>*1</sup>,
Jiangyong Huang<sup>1,2</sup>, Jingze Zhang<sup>1,2</sup>,
Siyuan Huang<sup>1,3</sup>
</p>

<p>
<sup>1</sup>State Key Laboratory of General Artificial Intelligence, BIGAI<br>
<sup>2</sup>Peking University &nbsp; <sup>3</sup>Delta Intelligence<br>
<sup>*</sup>Equal contribution
</p>

<p>
<a href="https://lmy1001.github.io/ICML26_LARA/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page"></a>
<a href="https://arxiv.org/pdf/2606.07100"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper"></a>
<a href="https://huggingface.co/MengyaLiu/LARA_full/tree/main"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-blue" alt="Model Checkpoints"></a>
</p>

<img src="assets/teaser.png" width="100%" alt="LARA teaser figure">

</div>

## Introduction

Vision-language-action (VLA) models enable robots to predict actions directly
from observations and language instructions, but their performance depends on
large-scale, high-quality data and is limited by the scarcity of real-world
robot action datasets. To facilitate VLA model learning with abundant
unlabeled human videos, Latent Action Models (LAMs) learn latent action
representations from visual dynamics to provide additional supervision for
VLA learning.

However, LAM and VLA are typically trained separately, leaving LAM ungrounded
during VLA training and VLA models constrained by frozen LAM representations.
To address these issues, we propose **Latent Action Representation Alignment
(LARA)**, a plug-and-play framework that jointly optimizes LAM and VLA through
representation alignment. This enables reciprocal benefits: LAMs learn with
action trajectories to avoid spurious visual changes, while VLAs are
regularized by forward dynamics learned within LAMs to reduce hallucinations
of functionally ineffective trajectories.

We demonstrate LARA's versatility and effectiveness for pre-training,
post-training enhancement of pre-trained VLA models, and LAM refinement,
achieving average improvements of approximately **10%**, **5%**, and **15%**
over three simulation benchmarks and one meticulously designed real-world
robotic manipulation benchmark.

## OpenPI Post-training Implementation

This repository contains the LIBERO post-training implementation of LARA for a
pretrained π0.5 model. It extends the PyTorch implementation of π0.5 in OpenPI
with supervision from a pretrained latent motion tokenizer.

The release intentionally provides one training entry point:

```bash
bash run_pi05_lara_libero.sh
```

## Training pipeline

The OpenPI post-training workflow uses two released components:

1. **Latent motion tokenizer pretraining** — run `run_tokenizer.sh` in
   `LARA_full`.
2. **π0.5 LIBERO post-training** — run
   `run_pi05_lara_libero.sh` in this repository.

Stage 1 is released in the separate `LARA_full` codebase. This
repository contains only the minimal Moto tokenizer modules needed to load the
Stage 1 checkpoint during Stage 2.

## Installation

The upstream OpenPI environment targets Ubuntu 22.04, Python 3.11, and an
NVIDIA CUDA GPU.

```bash
git submodule update --init --recursive third_party/libero
if ! git -C third_party/libero apply --reverse --check ../libero_release.patch 2>/dev/null; then
  git -C third_party/libero apply ../libero_release.patch
fi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The small tracked LIBERO patch removes an upstream interactive `pdb` stop from
the object-placement failure path while preserving the original exception.
It does not change benchmark observations, actions, tasks, or success logic.

The bundled OpenPI PyTorch implementation requires its Transformers
replacement files:

```bash
TRANSFORMERS_DIR="$(uv run python -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')"
cp -r src/openpi/models_pytorch/transformers_replace/* "${TRANSFORMERS_DIR}/"
```

## Required assets

Datasets and model weights are deliberately excluded. Copy the environment
template and replace every local path:

```bash
cp .env.example .env
set -a
source .env
set +a
```

The required variables are:

| Variable | Expected content |
| --- | --- |
| `OPENPI_BASE_MODEL_PATH` | Converted π0.5 PyTorch checkpoint directory containing `model.safetensors` |
| `OPENPI_ASSETS_DIR` | Parent directory of `libero/norm_stats.json` |
| `LARA_TOKENIZER_PATH` | Tokenizer checkpoint containing `config.yaml` or `config.json`, plus tokenizer weights |
| `LARA_VIT_MAE_PATH` | Local ViT-MAE-large image encoder |
| `LARA_LIBERO_DATASET` | LeRobot dataset ID or local dataset path; defaults to `physical-intelligence/libero` |
| `LARA_OUTPUT_DIR` | Checkpoint root; defaults to `./outputs` |

For π0.5 weight conversion and the original checkpoints, follow the
[OpenPI checkpoint documentation](https://github.com/Physical-Intelligence/openpi).
The tokenizer checkpoint and ViT-MAE weights must be obtained separately
under their respective terms.

The released tokenizer objective uses LPIPS with an ImageNet-pretrained VGG16
backbone. Torchvision downloads `vgg16-397923af.pth` on first use if it is not
already cached. For offline training, place that file under
`${TORCH_HOME}/hub/checkpoints/` and set `TORCH_HOME` before launching; the
weights are not included in this repository.

To match the released Stage 3 experiment, the LPIPS forward pass is evaluated
under `torch.no_grad()`. Its value remains part of the reported tokenizer loss,
but it does not backpropagate into the decoder; the reconstruction and VQ terms
provide the tokenizer gradients.

The released objective is
`action flow loss + 0.01 * latent-motion cosine loss + 0.01 * tokenizer reconstruction/VQ loss`.
These weights are assigned by the `pi05_lara_libero` configuration in
`src/openpi/training/config.py` and applied in `scripts/train_pytorch.py`.
The tokenizer's internal commitment, reconstruction, perceptual, and
hidden-reconstruction weights are read from the tokenizer checkpoint config.

Checkpoints are written to:

```text
${LARA_OUTPUT_DIR}/pi05_lara_libero/${LARA_EXP_NAME}/<STEP>/
```

## LIBERO evaluation

Initialize the evaluation submodule and client environment as described in
[the LIBERO evaluation guide](examples/libero/README.md). A released
checkpoint directory must contain `model.safetensors` and
`assets/libero/norm_stats.json`.

Set machine-local paths through environment variables:

```bash
export LARA_EVAL_CHECKPOINT=/path/to/pi05_lara_libero/checkpoint
export LARA_EVAL_OUTPUT_DIR=./eval_outputs/openpi_libero10
```

In terminal A, run:

```bash
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_lara_libero \
  --policy.dir "${LARA_EVAL_CHECKPOINT}"
```

In terminal B, run a 10-task smoke test with one rollout per task:

```bash
examples/libero/.venv/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_10 \
  --args.num-trials-per-task 1 \
  --args.seed 7 \
  --args.video-out-path "${LARA_EVAL_OUTPUT_DIR}/smoke_test_videos"
```

For the full LIBERO-10 evaluation, run:

```bash
examples/libero/.venv/bin/python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_10 \
  --args.num-trials-per-task 20 \
  --args.seed 7 \
  --args.video-out-path "${LARA_EVAL_OUTPUT_DIR}/videos"
```

## Citation

If you find LARA useful in your research, please cite our paper:

```bibtex
@article{liu2026lara,
  title={Lara: Latent action representation alignment for vision-language-action models},
  author={Liu, Mengya and Jia, Baoxiong and Huang, Jiangyong and Zhang, Jingze and Huang, Siyuan},
  journal={arXiv preprint arXiv:2606.07100},
  year={2026}
}
```

## Acknowledgements

This code is inherited from and extends
[Physical Intelligence's OpenPI](https://github.com/Physical-Intelligence/openpi).
The π0.5 model is described in
[π0.5: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054).

The latent motion tokenizer and Stage 1 pipeline build on
[Moto / Moto-GPT](https://github.com/TencentARC/Moto) and its paper,
[Moto: Latent Motion Token as the Bridging Language for Learning Robot
Manipulation from Videos](https://arxiv.org/abs/2412.04445).

Post-training and evaluation use
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and its paper,
[LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot
Learning](https://arxiv.org/abs/2306.03310).

We thank the authors of these projects for releasing their code and models.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [LICENSE](LICENSE),
[LICENSE_GEMMA.txt](LICENSE_GEMMA.txt), and [NOTICE](NOTICE) before
redistribution. Model weights and datasets may have additional licenses.
