import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .utils import pad_or_trim


class ASVspoofDataset(Dataset):
    def __init__(
        self,
        data_dir,
        protocol_path,
        input_length=64000,
        sample_rate=16000,
        subset_fraction=1.0,
        seed=1234,
    ):
        super(ASVspoofDataset, self).__init__()
        self.data_dir = data_dir
        self.input_length = input_length
        self.sample_rate = sample_rate
        self.subset_fraction = subset_fraction
        self.seed = seed

        self.utterances = []
        self.labels = []
        self._load_protocol(protocol_path)
        self._apply_subset_fraction()

    def _load_protocol(self, protocol_path):
        with open(protocol_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue

                file_name = parts[1]
                label_text = parts[-1].lower()
                if label_text == "bonafide":
                    label = 0
                elif label_text == "spoof":
                    label = 1
                else:
                    raise ValueError(
                        f"Unknown label '{parts[-1]}' in protocol file {protocol_path}: {line}"
                    )

                # Determine audio file path
                if "train" in protocol_path.lower():
                    audio_dir = os.path.join(self.data_dir, "ASVspoof2019_LA_train", "flac")
                elif "dev" in protocol_path.lower():
                    audio_dir = os.path.join(self.data_dir, "ASVspoof2019_LA_dev", "flac")
                elif "eval" in protocol_path.lower():
                    audio_dir = os.path.join(self.data_dir, "ASVspoof2019_LA_eval", "flac")
                else:
                    audio_dir = os.path.join(self.data_dir, "flac")

                file_path = os.path.join(audio_dir, f"{file_name}.flac")

                self.utterances.append(file_path)
                self.labels.append(label)

    def _apply_subset_fraction(self):
        if not 0 < self.subset_fraction <= 1:
            raise ValueError("subset_fraction must be in the range (0, 1]")
        if self.subset_fraction == 1 or len(self.utterances) == 0:
            return

        subset_size = max(1, int(len(self.utterances) * self.subset_fraction))
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(len(self.utterances), generator=generator)[:subset_size].tolist()
        indices.sort()
        self.utterances = [self.utterances[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        file_path = self.utterances[idx]
        label = self.labels[idx]

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            audio, sr = sf.read(file_path, dtype="float32")
        except Exception as exc:
            raise RuntimeError(f"Could not read audio file {file_path}: {exc}") from exc

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)

        waveform = torch.from_numpy(audio).unsqueeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        waveform = pad_or_trim(waveform, self.input_length)

        # Squeeze channel dimension: (1, L) -> (L,)
        waveform = waveform.squeeze(0)

        return waveform, label


def get_dataloaders(
    data_dir,
    batch_size,
    input_length=64000,
    sample_rate=16000,
    seed=1234,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    subset_fraction=1.0,
):
    torch.manual_seed(seed)

    train_protocol = os.path.join(
        data_dir,
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.train.trn.txt",
    )
    dev_protocol = os.path.join(
        data_dir,
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.dev.trl.txt",
    )

    # Load train dataset
    train_dataset = ASVspoofDataset(
        data_dir=data_dir,
        protocol_path=train_protocol,
        input_length=input_length,
        sample_rate=sample_rate,
        subset_fraction=subset_fraction,
        seed=seed,
    )

    # Load dev dataset
    dev_dataset = ASVspoofDataset(
        data_dir=data_dir,
        protocol_path=dev_protocol,
        input_length=input_length,
        sample_rate=sample_rate,
        subset_fraction=subset_fraction,
        seed=seed,
    )

    # Split dev: 90% for training augmentation, 10% for validation
    dev_size = len(dev_dataset)
    train_dev_size = int(0.9 * dev_size)
    val_size = dev_size - train_dev_size

    train_dev_subset, val_subset = torch.utils.data.random_split(
        dev_dataset, [train_dev_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    # Combine original train + 90% dev
    combined_train = torch.utils.data.ConcatDataset([train_dataset, train_dev_subset])

    use_persistent_workers = persistent_workers and num_workers > 0
    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
    )

    return train_loader, val_loader


def get_eval_dataloader(
    data_dir,
    batch_size,
    input_length=64000,
    sample_rate=16000,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    subset_fraction=1.0,
    seed=1234,
):
    eval_protocol = os.path.join(
        data_dir,
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.eval.trl.txt",
    )

    eval_dataset = ASVspoofDataset(
        data_dir=data_dir,
        protocol_path=eval_protocol,
        input_length=input_length,
        sample_rate=sample_rate,
        subset_fraction=subset_fraction,
        seed=seed,
    )

    use_persistent_workers = persistent_workers and num_workers > 0
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
    )

    return eval_loader
