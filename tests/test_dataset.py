import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from rawnet2.dataset import ASVspoofDataset, get_dataloaders, get_eval_dataloader


class TestASVspoofDatasetParsing:
    def test_bonafide_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                f.write("LA_0079 LA_E_9392005 - bonafide\n")

            os.makedirs(os.path.join(tmpdir, "flac"))
            audio = np.random.randn(16000).astype(np.float32)
            sf.write(os.path.join(tmpdir, "flac", "LA_E_9392005.flac"), audio, 16000)

            ds = ASVspoofDataset(tmpdir, proto, input_length=16000)
            assert len(ds) == 1

            wf, label = ds[0]
            assert label == 0  # bonafide -> 0

    def test_spoof_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                f.write("LA_0079 LA_E_9392053 A17 spoof\n")

            os.makedirs(os.path.join(tmpdir, "flac"))
            audio = np.random.randn(16000).astype(np.float32)
            sf.write(os.path.join(tmpdir, "flac", "LA_E_9392053.flac"), audio, 16000)

            ds = ASVspoofDataset(tmpdir, proto, input_length=16000)
            wf, label = ds[0]
            assert label == 1  # spoof -> 1

    def test_multiple_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                f.write("LA_0079 file1 - bonafide\n")
                f.write("LA_0079 file2 A01 spoof\n")
                f.write("LA_0034 file3 A17 spoof\n")

            os.makedirs(os.path.join(tmpdir, "flac"))
            for name in ["file1", "file2", "file3"]:
                audio = np.random.randn(16000).astype(np.float32)
                sf.write(os.path.join(tmpdir, "flac", f"{name}.flac"), audio, 16000)

            ds = ASVspoofDataset(tmpdir, proto, input_length=16000)
            assert len(ds) == 3

            assert ds[0][1] == 0  # bonafide
            assert ds[1][1] == 1  # spoof
            assert ds[2][1] == 1  # spoof

    def test_empty_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w"):
                pass

            os.makedirs(os.path.join(tmpdir, "flac"))
            ds = ASVspoofDataset(tmpdir, proto, input_length=16000)
            assert len(ds) == 0

    def test_invalid_line_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                f.write("LA_0079 file1 - bonafide\n")
                f.write("invalid line\n")  # Should be skipped (only 2 parts)
                f.write("LA_0079 file2 A01 spoof\n")

            os.makedirs(os.path.join(tmpdir, "flac"))
            for name in ["file1", "file2"]:
                audio = np.random.randn(16000).astype(np.float32)
                sf.write(os.path.join(tmpdir, "flac", f"{name}.flac"), audio, 16000)

            ds = ASVspoofDataset(tmpdir, proto, input_length=16000)
            assert len(ds) == 2

    def test_unknown_label_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                f.write("LA_0079 file1 A01 unknown\n")

            with pytest.raises(ValueError, match="Unknown label"):
                ASVspoofDataset(tmpdir, proto, input_length=16000)

    def test_subset_fraction_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                for i in range(100):
                    f.write(f"LA_0079 file{i} - bonafide\n")

            ds1 = ASVspoofDataset(
                tmpdir, proto, input_length=16000, subset_fraction=0.01, seed=1234
            )
            ds2 = ASVspoofDataset(
                tmpdir, proto, input_length=16000, subset_fraction=0.01, seed=1234
            )

            assert len(ds1) == 1
            assert ds1.utterances == ds2.utterances
            assert ds1.labels == ds2.labels

    def test_invalid_subset_fraction_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                f.write("LA_0079 file1 - bonafide\n")

            with pytest.raises(ValueError, match="subset_fraction"):
                ASVspoofDataset(tmpdir, proto, input_length=16000, subset_fraction=0)


class TestASVspoofDatasetAudioProcessing:
    def test_resample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                f.write("LA_0079 file1 - bonafide\n")

            os.makedirs(os.path.join(tmpdir, "flac"))
            # Create audio at 22050 Hz (needs resampling to 16000)
            audio = np.random.randn(22050).astype(np.float32)
            sf.write(os.path.join(tmpdir, "flac", "file1.flac"), audio, 22050)

            ds = ASVspoofDataset(tmpdir, proto, input_length=16000, sample_rate=16000)
            wf, _ = ds[0]
            assert wf.shape == (16000,)  # Should be resampled to target length

    def test_pad_short_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                f.write("LA_0079 file1 - bonafide\n")

            os.makedirs(os.path.join(tmpdir, "flac"))
            audio = np.random.randn(8000).astype(np.float32)  # 0.5 seconds
            sf.write(os.path.join(tmpdir, "flac", "file1.flac"), audio, 16000)

            ds = ASVspoofDataset(tmpdir, proto, input_length=64000)
            wf, _ = ds[0]
            assert wf.shape == (64000,)  # Should be padded to target length

    def test_crop_long_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                f.write("LA_0079 file1 - bonafide\n")

            os.makedirs(os.path.join(tmpdir, "flac"))
            audio = np.random.randn(128000).astype(np.float32)  # 8 seconds
            sf.write(os.path.join(tmpdir, "flac", "file1.flac"), audio, 16000)

            ds = ASVspoofDataset(tmpdir, proto, input_length=64000)
            wf, _ = ds[0]
            assert wf.shape == (64000,)  # Should be cropped to target length

    def test_waveform_is_1d(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                f.write("LA_0079 file1 - bonafide\n")

            os.makedirs(os.path.join(tmpdir, "flac"))
            audio = np.random.randn(16000).astype(np.float32)
            sf.write(os.path.join(tmpdir, "flac", "file1.flac"), audio, 16000)

            ds = ASVspoofDataset(tmpdir, proto, input_length=16000)
            wf, _ = ds[0]
            assert wf.dim() == 1  # Squeezed from (1, L) to (L,)

    def test_stereo_audio_is_converted_to_mono(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proto = os.path.join(tmpdir, "protocol.txt")
            with open(proto, "w") as f:
                f.write("LA_0079 file1 - bonafide\n")

            os.makedirs(os.path.join(tmpdir, "flac"))
            audio = np.random.randn(16000, 2).astype(np.float32)
            sf.write(os.path.join(tmpdir, "flac", "file1.flac"), audio, 16000)

            ds = ASVspoofDataset(tmpdir, proto, input_length=16000)
            wf, _ = ds[0]
            assert wf.shape == (16000,)


class TestDataLoaderSplit:
    def test_train_val_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            protocols_dir = os.path.join(tmpdir, "ASVspoof2019_LA_cm_protocols")
            os.makedirs(protocols_dir)

            # Train protocol: 10 samples
            train_proto = os.path.join(protocols_dir, "ASVspoof2019.LA.cm.train.trn.txt")
            with open(train_proto, "w") as f:
                for i in range(10):
                    f.write(f"SPK{i} train_{i} - bonafide\n")

            # Dev protocol: 10 samples
            dev_proto = os.path.join(protocols_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
            with open(dev_proto, "w") as f:
                for i in range(10):
                    f.write(f"SPK{i} dev_{i} - bonafide\n")

            # Create train audio files
            os.makedirs(os.path.join(tmpdir, "ASVspoof2019_LA_train", "flac"))
            for i in range(10):
                audio = np.random.randn(16000).astype(np.float32)
                sf.write(
                    os.path.join(tmpdir, "ASVspoof2019_LA_train", "flac", f"train_{i}.flac"),
                    audio,
                    16000,
                )

            # Create dev audio files
            os.makedirs(os.path.join(tmpdir, "ASVspoof2019_LA_dev", "flac"))
            for i in range(10):
                audio = np.random.randn(16000).astype(np.float32)
                sf.write(
                    os.path.join(tmpdir, "ASVspoof2019_LA_dev", "flac", f"dev_{i}.flac"),
                    audio,
                    16000,
                )

            train_loader, val_loader = get_dataloaders(
                data_dir=tmpdir,
                batch_size=2,
                input_length=16000,
                sample_rate=16000,
                seed=1234,
            )

            # Train = 10 (train set only, no dev leakage)
            # Val = 10 (full dev set, no speaker overlap)
            assert len(train_loader.dataset) == 10
            assert len(val_loader.dataset) == 10

    def test_eval_dataloader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            protocols_dir = os.path.join(tmpdir, "ASVspoof2019_LA_cm_protocols")
            os.makedirs(protocols_dir)

            eval_proto = os.path.join(protocols_dir, "ASVspoof2019.LA.cm.eval.trl.txt")
            with open(eval_proto, "w") as f:
                for i in range(5):
                    f.write(f"SPK{i} eval_{i} A07 spoof\n")

            os.makedirs(os.path.join(tmpdir, "ASVspoof2019_LA_eval", "flac"))
            for i in range(5):
                audio = np.random.randn(16000).astype(np.float32)
                sf.write(
                    os.path.join(tmpdir, "ASVspoof2019_LA_eval", "flac", f"eval_{i}.flac"),
                    audio,
                    16000,
                )

            eval_loader = get_eval_dataloader(
                data_dir=tmpdir,
                batch_size=2,
                input_length=16000,
                sample_rate=16000,
            )

            assert len(eval_loader.dataset) == 5
