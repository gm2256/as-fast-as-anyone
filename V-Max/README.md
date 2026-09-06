# AI-Challenge (Closed-loop Planning task) baseline

This repository is the training baseline for the challenge. It is a modified version of
[V-Max](https://github.com/valeoai/v-max) ([paper](https://arxiv.org/abs/2503.08388)) —
a JAX-based motion-planning framework built on the
[Waymax](https://github.com/waymo-research/waymax) simulator, providing closed-loop
RL/IL training pipelines (SAC, PPO, BC), network encoders, and driving metrics.
The modifications make the **rideflux challenge dataset** directly usable for training
and evaluation: dataset-preparation scripts (301-step records → WOMD-format 91-step
windows → Waymax sharded splits) plus the data-loading, observation, and metric
adaptations for this dataset.

The sections below cover the full pipeline: environment setup → dataset preparation →
training → evaluation → submission export.

## 1. Environment setup

Recommended system: **Ubuntu 24.04 LTS (x86_64)** with the **latest NVIDIA driver**
(the bundled CUDA 13 wheels require driver >= 580). No system CUDA/cuDNN installation
is needed — all CUDA user-space libraries are installed as Python wheels.

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone and install
git clone <this-repository>
cd V-Max
uv sync
```

`uv sync` creates `.venv` with Python 3.12 (pinned in `.python-version`, downloaded
automatically if not present) and installs the exact dependency versions recorded in
`uv.lock`, including JAX with the CUDA 13 runtime.

## 2. Dataset preparation

The challenge dataset is distributed as **301-step** TFRecords (past 150 / current 1 /
future 150 frames @ 10 Hz), laid out as `<root>/<site>/<date>/*.tfrecord`. Two scripts
turn it into a Waymax-loadable dataset.

### 2.1. Cut 301-step records into 91-step windows

Stock Waymax parses WOMD-format 91-step records (past 10 / current 1 / future 80).
`scripts/make_91f.py` cuts every 301-step record into 3 windows (starting at source
steps 0 / 100 / 200) and repacks each one as a standard WOMD tf_example:

```bash
uv run python scripts/make_91f.py /data/rideflux_301f /data/rideflux_91f
```

The output mirrors the input tree 1:1 (3 records per output file). The run is atomic
per file and resumable; append a number (e.g. `... make_91f.py <in> <out> 30`) for a
smoke run on 30 files first.

### 2.2. Build a Waymax sharded split

Waymax addresses datasets with sharded naming (`<name>.tfrecord@N`).
`scripts/make_waymax_shards.py` symlinks a dataset tree into that layout and writes a
`manifest.csv` (shard index -> source file):

```bash
uv run python scripts/make_waymax_shards.py /data/rideflux_91f /data/splits/rideflux_trainset_91f
```

Its last output line is the exact value to pass as `path_dataset`, e.g.

```
waymax path: /data/splits/rideflux_trainset_91f/rideflux_trainset_91f.tfrecord@85126
```

## 3. Training

Set `path_dataset` to the `...tfrecord@N` path printed above. `waymo_dataset=true` is
required for this dataset format (raw WOMD tf_examples, no precomputed SDC paths):

``` bash
CUDA_VISIBLE_DEVICES=0 uv run vmax/scripts/training/train.py  \
    algorithm=sac \
    network/encoder=lq \
    total_timesteps=25_000_000 \
    algorithm.learning_rate=1e-4 \
    algorithm.buffer_size=1_000_000 \
    algorithm.learning_start=50_000 \
    'algorithm.network.policy.layer_sizes=[256,64,32]' \
    'algorithm.network.value.layer_sizes=[256,64,32]' \
    observation_config.objects.num_closest_objects=16 \
    waymo_dataset=true \
    path_dataset=/data/splits/rideflux_trainset_91f/rideflux_trainset_91f.tfrecord@85126 \
    name_run=abcdef
```

Checkpoints and TensorBoard logs go to `runs/<run_name>/`.

## 4. Evaluation

`--path_model` is the `name_run` of the training run. `--path_dataset` accepts the
same `@N` sharded naming, or the path of a plain single `.tfrecord` file. Results are
written next to the model as `evaluation_episodes.csv` / `evaluation_results.txt`:

``` bash
CUDA_VISIBLE_DEVICES=0 uv run python vmax/scripts/evaluate/evaluate.py \
    --waymo_dataset=true \
    --path_dataset=/data/splits/rideflux_valset_91f/rideflux_valset_91f.tfrecord@N \
    --sdc_actor=ai \
    --path_model=abcdef \
    --batch_size=64
```

## 5. Exporting policy weights for submission

Checkpoints under `runs/<run_name>/model/` pickle the full network state (policy +
value networks, as jax arrays) and can only be unpickled where the `vmax` package is
importable. A challenge submission needs the policy alone, as a plain dict of numpy
arrays (`weights.pkl`):

``` bash
uv run python scripts/export_policy_weights.py \
    runs/<run_name>/model/model_final.pkl <submission_dir>/weights.pkl
```
