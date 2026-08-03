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

## Quick Start

### Installation

Python 3.10, a CUDA-capable PyTorch environment, and an NVIDIA GPU are
recommended.

```bash
pip install -e .
```

FlashAttention is optional. Install it in a compatible CUDA build environment:

```bash
INSTALL_FLASH_ATTN=1 bash install.sh
```

Without FlashAttention, the RADIO backbone uses the standard timm attention
implementation.

Copy the environment template and replace the model and output placeholders:

```bash
cp .env.example .env
set -a
source .env
set +a
```

### Model Checkpoints

Released LARA model checkpoints are hosted on
[Hugging Face](https://huggingface.co/MengyaLiu/LARA_full/tree/main). Datasets,
the GR00T N1.5 base model, and the ViT-MAE image encoder remain external
assets and are not included in this source release.

## Training

LARA_full provides the complete three-stage training pipeline:

1. pre-train the latent motion tokenizer;
2. jointly pre-train the LARA action DiT and latent motion tokenizer;
3. post-train the Stage 2 LARA model on downstream robot demonstrations.

Each training stage reads one `data_1.dataset_path` directly from its selected
data YAML. The release does not use a data-root environment-variable mapping.

### Stage 1: Latent Motion Tokenizer Pre-training

Set the image encoder, output directory, and GPU count:

```bash
export LARA_VIT_MAE_PATH=/path/to/vit-mae-large
export LARA_TOKENIZER_OUTPUT_DIR=./outputs/tokenizer
export LARA_NUM_GPUS=4
```

Set the dataset path and modality keys in
`moto/configs/data/tokenizer_pretrain_data.yaml`, then launch:

```bash
bash run_tokenizer.sh
```

Additional Hydra overrides are forwarded unchanged:

```bash
bash run_tokenizer.sh max_steps=10000 batch_size=64
```

Select one saved tokenizer checkpoint for Stage 2:

```bash
export LARA_TOKENIZER_PATH="${LARA_TOKENIZER_OUTPUT_DIR}/checkpoint-<STEP>"
```

### Stage 2: LARA Joint Pre-training

Set the GR00T N1.5 base model, Stage 1 tokenizer, image encoder, and output
directory:

```bash
export LARA_BASE_MODEL_PATH=/path/to/GR00T-N1.5-3B
export LARA_TOKENIZER_PATH=/path/to/tokenizer/checkpoint
export LARA_VIT_MAE_PATH=/path/to/vit-mae-large
export LARA_PRETRAIN_OUTPUT_DIR=./outputs/lara_pretrain
export LARA_NUM_GPUS=8
```

Set the dataset path and modalities in
`lara_full/config/data/mani_test_data_all.yaml`, then launch:

```bash
bash run_lara_full_pretrain.sh
```

Additional Hydra overrides are supported:

```bash
bash run_lara_full_pretrain.sh batch_size=32 max_steps=50000
```

### Stage 3: Downstream Post-training on LIBERO-10

Stage 3 continues from the trained Stage 2 LARA checkpoint while retaining the
GR00T N1.5 architecture. Set the checkpoint hand-off and output paths:

```bash
export LARA_STAGE2_CHECKPOINT=/path/to/lara_pretrain/checkpoint-<STEP>
export LARA_VIT_MAE_PATH=/path/to/vit-mae-large
export LARA_POSTTRAIN_OUTPUT_DIR=./outputs/lara_posttrain_libero_10
export LARA_NUM_GPUS=4
```

The launcher loads the Stage 2-tuned tokenizer from
`${LARA_STAGE2_CHECKPOINT}/latent_motion_tokenizer` by default. For a legacy
checkpoint with a different layout, set `LARA_POSTTRAIN_TOKENIZER_PATH`
explicitly.

Set the single LIBERO-10 LeRobot dataset path in
`lara_full/config/data/libero_10.yaml`, then launch:

```bash
bash run_libero_10.sh
```

Additional Hydra overrides are forwarded unchanged:

```bash
bash run_libero_10.sh batch_size=32 max_steps=10000
```

## Evaluation

### LIBERO-10

The released evaluation uses the GR00T N1.5 inference stack. The
[model server](scripts/inference_service.py) and
[LIBERO client](examples/libero/run_libero_eval.py) run in separate
environments so that model and MuJoCo dependencies do not conflict.

In the model-server environment, install this repository with the evaluation
transport dependencies:

```bash
pip install -e '.[libero]'
```

In the simulator environment, install
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and the lightweight
client dependencies:

```bash
pip install imageio[ffmpeg] msgpack pyzmq tqdm tyro
```

Point to a concrete Stage 3 checkpoint directory, not the Stage 3 output root
or the Stage 2 input checkpoint:

```bash
export LARA_EVAL_CHECKPOINT="${LARA_POSTTRAIN_OUTPUT_DIR}/checkpoint-<STEP>"
export LARA_EVAL_OUTPUT_DIR=./eval_outputs/lara_full_libero10
```

The checkpoint must contain `config.json` and
`experiment_cfg/metadata.json`. The latter is written during Stage 3 training
and must contain the `libero` normalization statistics.

In terminal A, serve the checkpoint:

```bash
python scripts/inference_service.py \
  --model-path "${LARA_EVAL_CHECKPOINT}" \
  --host 0.0.0.0 \
  --port 5555 \
  --denoising-steps 4
```

In terminal B, run the LIBERO simulator client:

```bash
export MUJOCO_GL=egl

python examples/libero/run_libero_eval.py \
  --host 127.0.0.1 \
  --port 5555 \
  --task-suite-name libero_10 \
  --num-trials-per-task 1 \
  --seed 0 \
  --output-dir "${LARA_EVAL_OUTPUT_DIR}"
```

Use `--num-trials-per-task 20` for the standard 200-episode evaluation. For a
short visual smoke test, add `--task-id 0 --save-video` while keeping one
trial. The client saves aggregate and per-episode results to `results.json`.

The released protocol uses two 256 × 256 camera views rotated by 180 degrees,
a 16-step action horizon, four denoising steps, ten simulator stabilization
steps, at most 1,000 policy steps, and action index 1 from each predicted
chunk. See the [detailed LIBERO guide](examples/libero/README.md) for more
information.

## Release Verification

Run the lightweight release checks without creating pytest cache files:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_release.py tests/test_libero_eval.py
```

These checks guard the single-path launcher contract, private-path and debug
residue, obsolete tokenizer code, the tokenizer embed supervision target, and
the LIBERO transport/action protocol.

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

The latent motion tokenizer and Stage 1 pipeline build on
[Moto / Moto-GPT](https://github.com/TencentARC/Moto) and its paper,
[Moto: Latent Motion Token as the Bridging Language for Learning Robot
Manipulation from Videos](https://arxiv.org/abs/2412.04445).

The VLA backbone, action flow matching, data transforms, and Stage 2/3
infrastructure build on
[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), specifically the
GR00T N1.5 code path, and the paper
[GR00T N1: An Open Foundation Model for Generalist Humanoid
Robots](https://arxiv.org/abs/2503.14734).

We thank the authors of these projects for releasing their code and models.

## License

This repository is distributed under the terms in [LICENSE](LICENSE).
Third-party provenance is summarized in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); individual source headers
remain authoritative. Model weights and datasets may have additional license
terms.
