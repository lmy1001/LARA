# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

LARA checkpoints in this repository must be loaded with the
`pi05_lara_libero` config. The upstream OpenPI config named `pi05_libero` is a
different experiment and must not be used for a LARA checkpoint.

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive third_party/libero
if ! git -C third_party/libero apply --reverse --check ../libero_release.patch 2>/dev/null; then
  git -C third_party/libero apply ../libero_release.patch
fi
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# Mount the host checkpoint parent at /openpi_assets in the server container.
export OPENPI_DATA_HOME=/path/to/checkpoint-parent
export LARA_EVAL_CHECKPOINT_IN_CONTAINER=/openpi_assets/checkpoint-directory
export SERVER_ARGS="policy:checkpoint --policy.config pi05_lara_libero --policy.dir ${LARA_EVAL_CHECKPOINT_IN_CONTAINER}"
export CLIENT_ARGS="--args.task-suite-name libero_10 --args.num-trials-per-task 20"

docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx docker compose -f examples/libero/compose.yml up --build
```

Customize `SERVER_ARGS` with options from `scripts/serve_policy.py` and
`CLIENT_ARGS` with options from `examples/libero/main.py`. Do not use
`SERVER_ARGS="--env LIBERO"` for LARA: that selects the upstream default
checkpoint instead of the supplied LARA checkpoint.

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH="${PWD}/third_party/libero${PYTHONPATH:+:${PYTHONPATH}}"

# Run a 10-task smoke test with one rollout per task.
MUJOCO_GL=egl python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_10 \
  --args.num-trials-per-task 1 \
  --args.video-out-path ./eval_outputs/openpi_libero10/videos_smoke

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py --args.task-suite-name libero_10
```

Terminal window 2:

```bash
# Run the LARA checkpoint server.
export LARA_EVAL_CHECKPOINT=/path/to/pi05_lara_libero/checkpoint
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_lara_libero \
  --policy.dir "${LARA_EVAL_CHECKPOINT}"
```

## Results

The table below is retained as an upstream OpenPI reference. It uses the
upstream `pi05_libero` checkpoint and config, not the LARA checkpoint or
`pi05_lara_libero` config documented above.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
