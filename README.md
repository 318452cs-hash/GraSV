# GraSV

GraSV is a graph-based structural variant inference pipeline for long-read data.
It can run from either an indexed BAM or a precomputed `signatures.pkl`.

## Install

Recommended clean conda setup:

```bash
conda create -n grasv python=3.10 -y
conda activate grasv

# Keep this environment isolated from ~/.local Python packages.
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate grasv

python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=1.23,<2"
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.2.2+cpu"
python -m pip install -e ".[test]"
```

For a CUDA-enabled PyTorch build, install the matching `torch` package for your
server before `python -m pip install -e ".[test]"`. Avoid plain `pip`; use
`python -m pip` from the active environment. Keep NumPy below 2 for
`torch==2.2.2` compatibility.

## Quick Check

After installation, verify the CLI and bundled model paths:

```bash
grasv infer --help
```

Source-tree entry before installation:

```bash
PYTHONPATH=src python3 -m grasv.cli infer --help
```

## BAM Inference

```bash
grasv infer \
  --platform ont \
  --bam-path /path/to/sample.bam \
  --output-dir outputs/sample_ont \
  --processes 8
```

The BAM must be indexed. If `--global-coverage` is omitted, GraSV estimates it
from the BAM index and sampled read lengths.
`--processes` enables chromosome-level clustering parallelism and is also used
when signatures are extracted directly from BAM.

## Logging

GraSV writes stage-level logs to stderr by default. Use `--log-level DEBUG` for
chromosome-level clustering progress, or `--log-file` to keep a persistent run
log:

```bash
grasv infer \
  --platform ont \
  --bam-path /path/to/sample.bam \
  --output-dir outputs/sample_ont \
  --processes 8 \
  --log-level DEBUG \
  --log-file outputs/sample_ont/grasv.log
```

## Model Files

Default checkpoints are stored in `models/`:

- `grasv_encoder.pt`
- `grasv_scorer.pt`

The downstream checkpoint is a single-head CNN scorer used only for call
filtering. This release reports site-level SV calls only.

To use external checkpoints:

```bash
grasv infer \
  --platform ont \
  --bam-path /path/to/sample.bam \
  --model-path /path/to/encoder.pt \
  --cluster-scorer-path /path/to/scorer.pt \
  --output-dir outputs/sample_custom
```

Public GraSV inference expects the downstream checkpoint to be `cnn_scorer_v1`.

You can also set `GRASV_MODEL_DIR=/path/to/models`.

## Code Layout

- `src/grasv/pipeline.py`: current GraSV inference orchestration.
- `src/grasv/config.py`: coverage-aware graph and postfilter presets.
- `src/grasv/graph.py`: type-grouped affinity graph construction and connectivity clustering.
- `src/grasv/scorer.py`: CNN scorer loading/application.
- `src/grasv/data.py`, `signature_extraction.py`, `signature_features.py`, `calling.py`: BAM/signature IO, feature construction, call generation, and VCF export support.

The repository intentionally excludes historical batch scripts, local absolute
paths, full evaluation outputs, and large intermediate training artifacts.

## Notes On Evaluation

GraSV selects graph/postfilter parameters from platform and coverage. Treat
coverage-specific results carefully. Evaluation is intentionally not included
in the GraSV CLI; users should evaluate output VCFs with their own held-out
truth sets and preferred external tools.

Recommended evaluation practice: use held-out samples when possible; if only
one sample is available, separate model development and reporting clearly; keep
final truth sets out of threshold and preset tuning; and report whether default
presets, validation-selected thresholds, or manual per-coverage settings were
used.
