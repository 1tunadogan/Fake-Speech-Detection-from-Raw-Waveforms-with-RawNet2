# Agent Notes

## Build Commands

```bash
# Install dependencies
uv sync

# Run training
uv run python -m rawnet2.train --config config.yaml

# Run evaluation
uv run python -m rawnet2.evaluate --config config.yaml --checkpoint weights/best.pth

# Lint
uv run ruff check src/
uv run ruff format src/
```

## Key Architecture Decisions

- **SincConv**: Fixed (non-learned) sinc filters initialized with Mel/Inverse-Mel/Linear scale.
- **FMS**: Applied after every ResidualBlock via `x * scale + scale` (additive + multiplicative).
- **GRU**: 3 layers, 1024 hidden units (matches original implementation config).
- **Loss**: Weighted CrossEntropy with `weight=[1.0, 9.0]` for spoof class.
- **Data Split**: Original train + 90% of dev for training, remaining 10% of dev for validation.
- **Input**: Raw waveforms padded/trimmed to 64000 samples (~4s @ 16kHz).

## PyTorch Version

- torch>=2.11.0
- torchaudio>=2.11.0
- Uses `F.conv1d`, `nn.GRU`, `nn.LeakyReLU`, `nn.BatchNorm1d`, `nn.MaxPool1d`

## W&B Integration

- `train.py`: `wandb.init()` + `run.watch()` + `run.log()` per batch/epoch + `run.log_model()` for best checkpoint
- `evaluate.py`: `wandb.init(job_type="eval")` + `run.use_model()` for artifact loading + `run.log()` for eval metrics

## W&B Setup

- Get API key from: https://wandb.ai/authorize
- Login: `wandb login` (interactive) or `set WANDB_API_KEY=your-key` (Windows)
- Project: `rawnet2-antispoofing` (configurable in `config.yaml`)
- Offline mode for tests without internet: `mode: "offline"` in config

## Dataset

- ASVspoof 2019 LA logical access (LA) track
- Protocol files: `ASVspoof2019.LA.cm.{train,dev,eval}.trl.txt`
- Audio: `.flac` files in `data/LA/ASVspoof2019_LA_{train,dev,eval}/flac/`
