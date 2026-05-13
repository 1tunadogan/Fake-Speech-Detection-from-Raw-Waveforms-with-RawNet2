import os

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
    ):
        super(ASVspoofDataset, self).__init__()
        self.data_dir = data_dir
        self.input_length = input_length
        self.sample_rate = sample_rate

        self.utterances = []
        self.labels = []
        self._load_protocol(protocol_path)

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
                key = parts[3]

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

                # Bonafide (authentic): key="-", Spoof (attack): key="A01"-"A06", etc.
                label = 0 if key == "-" else 1

                self.utterances.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        file_path = self.utterances[idx]
        label = self.labels[idx]

        audio, sr = sf.read(file_path, dtype="float32")
        waveform = torch.from_numpy(audio).unsqueeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )

        waveform = pad_or_trim(waveform, self.input_length)

        # Squeeze channel dimension: (1, L) -> (L,)
        waveform = waveform.squeeze(0)

        return waveform, label


def get_dataloaders(data_dir, batch_size, input_length=64000, sample_rate=16000, seed=1234):
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
    )

    # Load dev dataset
    dev_dataset = ASVspoofDataset(
        data_dir=data_dir,
        protocol_path=dev_protocol,
        input_length=input_length,
        sample_rate=sample_rate,
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

    train_loader = DataLoader(
        combined_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def get_eval_dataloader(data_dir, batch_size, input_length=64000, sample_rate=16000):
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
    )

    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return eval_loader
