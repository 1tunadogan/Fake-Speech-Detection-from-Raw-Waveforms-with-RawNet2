# Fake Speech Detection from Raw Waveforms with RawNet2

End-to-end fake speech detection using RawNet2 on raw audio waveforms.
Based on the paper **"End-to-end anti-spoofing with RawNet2"** (Tak et al., ICASSP 2021).

## Project Structure

```
.
├── config.yaml              # All hyperparameters and W&B settings
├── pyproject.toml           # uv project configuration
├── scripts/
│   └── download_dataset.py  # Dataset downloader (Kaggle -> data/LA)
├── src/
│   └── rawnet2/
│       ├── __init__.py
│       ├── model.py         # RawNet2 architecture (SincConv + FMS + GRU)
│       ├── dataset.py       # ASVspoof 2019 LA data loading
│       ├── utils.py         # EER, min t-DCF, device helpers
│       ├── train.py         # Training script with W&B tracking
│       └── evaluate.py      # Evaluation script with W&B artifact loading
├── data/                    # ASVspoof 2019 LA dataset (not tracked)
│   └── LA/
│       ├── ASVspoof2019_LA_cm_protocols/
│       ├── ASVspoof2019_LA_train/flac/
│       ├── ASVspoof2019_LA_dev/flac/
│       └── ASVspoof2019_LA_eval/flac/
└── weights/                 # Saved checkpoints (not tracked)
```

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Dataset

This project uses the **ASVspoof 2019 LA** (Logical Access) subset.

### Option A: Automatic Download (Recommended)

```bash
uv run python scripts/download_dataset.py
```

This downloads the LA-only subset (~7.66 GB, 122k files) from Kaggle and
auto-creates a junction at `data/LA`. No config changes needed.

**Prerequisite:** Kaggle API credentials
- Get token: https://www.kaggle.com/settings -> API -> Create New Token
- Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows)
- Or set env vars: `KAGGLE_USERNAME` + `KAGGLE_KEY`

### Option B: Manual Download

If you prefer manual download, get the LA dataset from either source:

- **Kaggle** (recommended): https://www.kaggle.com/datasets/anishsarkar22/asvpoof-2019-dataset-la
- **Official source**: https://datashare.ed.ac.uk/handle/10283/3336

Extract and place under `data/LA/` with this exact structure:

```
data/LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
└── ASVspoof2019_LA_eval/flac/
```

> `config.yaml` already points to `data/LA` via `data.data_dir`. No edits needed.

## Full End-to-End Run (Train + Eval + W&B Online)

Run this sequence once for a full pipeline execution:

```bash
# 1) Install dependencies
uv sync

# 2) Login to W&B (one-time per machine/session as needed)
uv run wandb login --relogin

# 3) Ensure cloud sync is enabled
uv run wandb online

# 4) Download dataset (if not already present)
uv run python scripts/download_dataset.py

# 5) Train
uv run python -m rawnet2.train --config config.yaml

# 6) Evaluate (local best checkpoint)
uv run python -m rawnet2.evaluate --config config.yaml --checkpoint weights/best.pth
```

Notes:
- Device is selected automatically in this order: **CUDA -> MPS -> CPU**.
- `model.sinc_scale` accepts lowercase values only: `mel`, `inverse-mel`, `linear`.
- With `wandb.mode: "online"` in `config.yaml`, runs sync to W&B cloud automatically.

## Training

```bash
# Default config (S1: Mel-scale sinc filters)
uv run python -m rawnet2.train --config config.yaml

# S2: Inverse-Mel scale
# Edit config.yaml: model.sinc_scale: "inverse-mel"
uv run python -m rawnet2.train --config config.yaml

# S3: Linear scale
# Edit config.yaml: model.sinc_scale: "linear"
uv run python -m rawnet2.train --config config.yaml
```

## Evaluation

```bash
# Evaluate using local checkpoint
uv run python -m rawnet2.evaluate --config config.yaml --checkpoint weights/best.pth

# Evaluate using W&B artifact (auto-load best model)
uv run python -m rawnet2.evaluate --config config.yaml
```

## Lint

```bash
uv run ruff check src/
uv run ruff format src/
```

## Architecture Details

| Layer | Input | Output |
|-------|-------|--------|
| SincConv (fixed, 128 filters) + MaxPool(3) | (B, 64000) | (B, 128, 21290) |
| ResBlock × 2 [128→128] + FMS | (B, 128, 21290) | (B, 128, 2365) |
| ResBlock × 4 [128→512] + FMS | (B, 128, 2365) | (B, 512, 29) |
| GRU (3×1024) | (B, 29, 512) | (B, 1024) |
| FC (1024→1024→2) | (B, 1024) | (B, 2) |

## W&B Integration

Experiments are tracked via Weights & Biases. Configure in `config.yaml` under the `wandb` section:

```yaml
wandb:
  project: "rawnet2-antispoofing"
  mode: "online"     # online | offline | disabled
  log_model: true    # Save best checkpoint as artifact
```

Default naming behavior:
- Train run name: `RawNet2-{sinc_scale}-train`
- Eval run name: `RawNet2-{sinc_scale}-eval`
- Model artifact name: `RawNet2-{sinc_scale}`

If you set `wandb.name` in `config.yaml`, that explicit value overrides the default run naming.

## References

- Tak et al., "End-to-end anti-spoofing with RawNet2", ICASSP 2021
- Jung et al., "RawNet2: Improved end-to-end speaker verification", INTERSPEECH 2020
- ASVspoof 2019 LA Database: https://datashare.ed.ac.uk/
