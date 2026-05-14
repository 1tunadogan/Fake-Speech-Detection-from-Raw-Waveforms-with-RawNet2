import numpy as np
import pytest
import torch

from rawnet2.utils import (
    compute_eer,
    compute_min_tdcf,
    get_device,
    keras_lr_decay,
    pad_or_trim,
    set_seed,
)


class TestPadOrTrim:
    def test_pad_short(self):
        x = torch.randn(1, 32000)
        out = pad_or_trim(x, 64000)
        assert out.shape == (1, 64000)

    def test_crop_long(self):
        x = torch.randn(1, 128000)
        out = pad_or_trim(x, 64000)
        assert out.shape == (1, 64000)

    def test_exact_length(self):
        x = torch.randn(1, 64000)
        out = pad_or_trim(x, 64000)
        assert out.shape == (1, 64000)
        assert torch.equal(x, out)

    def test_tiling(self):
        # Short audio should be tiled/repeated
        x = torch.randn(1, 16000)
        out = pad_or_trim(x, 64000)
        assert out.shape == (1, 64000)
        # First 16000 samples should repeat
        assert torch.equal(out[:, :16000], out[:, 16000:32000])

    def test_mono_channel(self):
        x = torch.randn(1, 32000)
        out = pad_or_trim(x, 64000)
        assert out.shape == (1, 64000)

    def test_empty_waveform_raises(self):
        x = torch.empty(1, 0)
        with pytest.raises(ValueError, match="empty waveform"):
            pad_or_trim(x, 64000)


class TestComputeEER:
    def test_perfect_separation(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        eer = compute_eer(scores, labels)
        assert eer < 1.0

    def test_random_guessing(self):
        np.random.seed(42)
        scores = np.random.rand(10000)
        labels = np.random.randint(0, 2, 10000)
        eer = compute_eer(scores, labels)
        # Random guessing should give EER around 50%
        assert 40 < eer < 60

    def test_all_bonafide(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 0, 0])
        # Single class: EER is undefined
        eer = compute_eer(scores, labels)
        assert np.isnan(eer)

    def test_all_spoof(self):
        scores = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 1])
        eer = compute_eer(scores, labels)
        assert np.isnan(eer)


class TestComputeMinTDCF:
    def test_perfect_separation(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        tdcf = compute_min_tdcf(scores, labels)
        assert tdcf < 0.01

    def test_random_guessing(self):
        np.random.seed(42)
        scores = np.random.rand(10000)
        labels = np.random.randint(0, 2, 10000)
        tdcf = compute_min_tdcf(scores, labels)
        # Random guessing should give moderate to high t-DCF
        assert tdcf > 0.01

    def test_default_costs(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        tdcf = compute_min_tdcf(scores, labels)
        # With P_target=0.05, C_miss=1, C_fa=10
        # Perfect separation should give very low t-DCF
        assert tdcf < 0.01

    def test_all_bonafide(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        labels = np.array([0, 0, 0, 0])
        tdcf = compute_min_tdcf(scores, labels)
        assert np.isnan(tdcf)

    def test_all_spoof(self):
        scores = np.array([0.6, 0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1, 1])
        tdcf = compute_min_tdcf(scores, labels)
        assert np.isnan(tdcf)


class TestGetDevice:
    def test_cuda_priority_over_mps(self, monkeypatch):
        class _FakeMPS:
            @staticmethod
            def is_built():
                return True

            @staticmethod
            def is_available():
                return True

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends, "mps", _FakeMPS(), raising=False)

        device = get_device()
        assert device.type == "cuda"

    def test_mps_when_no_cuda(self, monkeypatch):
        class _FakeMPS:
            @staticmethod
            def is_built():
                return True

            @staticmethod
            def is_available():
                return True

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends, "mps", _FakeMPS(), raising=False)

        device = get_device()
        assert device.type == "mps"

    def test_cpu_fallback(self, monkeypatch):
        class _FakeMPS:
            @staticmethod
            def is_built():
                return True

            @staticmethod
            def is_available():
                return False

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends, "mps", _FakeMPS(), raising=False)

        device = get_device()
        assert device.type == "cpu"

    def test_is_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_different_seeds(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(123)
        b = torch.randn(10)
        assert not torch.allclose(a, b)

    def test_numpy_reproducibility(self):
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        assert np.allclose(a, b)


class TestKerasLRDecay:
    def test_decreases(self):
        lr0 = keras_lr_decay(0)
        lr1 = keras_lr_decay(1000)
        assert lr1 < lr0

    def test_step_0(self):
        lr = keras_lr_decay(0)
        assert lr == 1.0

    def test_positive(self):
        lr = keras_lr_decay(10000, decay=0.0001)
        assert lr > 0
        assert lr < 1.0
