# Stage 3 LARA_full LIBERO-10 evaluation

The Stage 3 LIBERO checkpoint uses the GR00T N1.5 inference stack. The model
server is implemented in `scripts/inference_service.py`, and the simulator
client is implemented in `examples/libero/run_libero_eval.py`. Run them in
separate environments so that CUDA model dependencies do not conflict with
LIBERO/MuJoCo.

Install this repository with the evaluation transport on the server side:

```bash
pip install -e '.[libero]'
```

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) in the
client environment, then install the lightweight client requirements:

```bash
pip install imageio[ffmpeg] msgpack pyzmq tqdm tyro
```

From the repository root, terminal A serves a concrete Stage 3 checkpoint.
Point to `checkpoint-<STEP>`, not the parent output directory:

```bash
export LARA_EVAL_CHECKPOINT=/path/to/lara_posttrain_libero_10/checkpoint-<STEP>

python scripts/inference_service.py \
  --model-path "${LARA_EVAL_CHECKPOINT}" \
  --host 0.0.0.0 \
  --port 5555 \
  --denoising-steps 4
```

After the server prints that it is listening, terminal B runs one rollout for
each of the ten LIBERO-10 tasks:

```bash
export MUJOCO_GL=egl
export LARA_EVAL_OUTPUT_DIR=./eval_outputs/lara_full_libero10_smoke

python examples/libero/run_libero_eval.py \
  --host 127.0.0.1 \
  --port 5555 \
  --task-suite-name libero_10 \
  --num-trials-per-task 1 \
  --seed 0 \
  --output-dir "${LARA_EVAL_OUTPUT_DIR}"
```

Change `--num-trials-per-task 1` to `20` for the standard 200-episode run.
Use `--task-id 0` for a one-task debugging run and add `--save-video` to keep
rollout videos. The client saves aggregate and per-episode results in
`results.json` under the selected output directory.

This client matches the released checkpoint protocol: two 256 x 256 camera
views rotated by 180 degrees, a 16-step action horizon, four denoising steps,
ten simulator stabilization steps, up to 1,000 policy steps, and action index
1 from every predicted chunk. It queries the policy at every environment step.
Do not evaluate this checkpoint through the GR00T N1.6 pipeline.
