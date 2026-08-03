# DiffusionJEPA Submission Code

This directory contains the minimal training and evaluation code for the four
main tasks: Cube, Push-T, Reacher, and TwoRoom.

## Reproducibility Protocol

The submission pipeline always runs the following stages:

```text
raw HDF5
  -> deterministic 80/20 episode-level train/test split
  -> planner tuples built from the train split only
  -> K-means-nearest anchors built from train tuples only
  -> diffusion planner training
  -> closed-loop evaluation on the test split only
```

There is no raw-dataset bypass in this package. The split seed defaults to 42.
All transitions from one episode remain on the same side of the split.

## K-means-nearest Anchors

`diffusion/anchors.py` fits K-means to flattened action chunks, then replaces
every centroid with the nearest real action chunk assigned to that cluster.
Consequently, every stored anchor is an observed training action sequence, not
an averaged centroid. Anchor bundles record:

- `fit_method: kmeans_nearest_real_sample`
- selected planner-dataset row indices
- mean and maximum centroid-to-real L2 distance
- K-means inertia and iteration count

The common `planners/build_action_anchors.py` entry point uses this algorithm
for every task. The resolved configuration exposes the rule as
`anchors.selection: kmeans-nearest`.

## Layout

```text
config/diffusion/       end-to-end pipeline configuration
config/eval/            closed-loop evaluation configuration
diffusion/              anchor, model, training, policy, and pipeline code
planners/               planner dataset and compatibility modules
evaluation/             task registration and trajectory metrics
scripts/run_pipeline.py end-to-end Hydra entry point
scripts/split_hdf5_by_episode.py
eval.py                  evaluation entry point
train_diffusion_planner.py
tests/                   protocol regression tests
```

## Setup

Use Python 3.10 or newer. Install PyTorch for the target CUDA version first,
then install the remaining dependencies:

```bash
cd submission
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default data root is `/data/lewm`. Override it without editing YAML:

```bash
export LEWM_DATA_ROOT=/path/to/data
```

Expected raw files and world-model checkpoints are declared in
`config/diffusion/task/*.yaml`.

## Run

Run a complete task pipeline:

```bash
cd submission
python -u scripts/run_pipeline.py task=cube
python -u scripts/run_pipeline.py task=pusht
python -u scripts/run_pipeline.py task=reacher
python -u scripts/run_pipeline.py task=tworoom
```

Inspect all resolved paths and stage commands without writing outputs:

```bash
python -u scripts/run_pipeline.py task=reacher pipeline.dry_run=true pipeline.device=cpu
```

The split outputs follow these conventions:

```text
<task-root>/splits/<dataset>_train/...h5
<task-root>/splits/<dataset>_test/...h5
```

## Individual Stages

Build anchors from an existing planner tuple bundle:

```bash
python -m diffusion.anchor_builder \
  --mode build \
  --dataset-path /path/to/planner_dataset.pt \
  --output-path /path/to/action_anchors_k128_kmeans_nearest.pt \
  --num-anchors 128 \
  --selection kmeans-nearest \
  --max-samples 200000 \
  --seed 42
```

Run protocol tests:

```bash
python -m unittest discover -s tests -v
```
