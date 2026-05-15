import random

import numpy as np
import torch
from sklearn.metrics import roc_curve


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_backend is not None
        and hasattr(mps_backend, "is_available")
        and hasattr(mps_backend, "is_built")
        and mps_backend.is_built()
        and mps_backend.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def pad_or_trim(waveform, target_length):
    """Pad short waveforms by tiling, crop long ones.

    Args:
        waveform: Tensor of shape (channels, time)
        target_length: Target number of samples

    Returns:
        Tensor of shape (channels, target_length)
    """
    current_length = waveform.shape[1]
    if current_length == 0:
        raise ValueError("Cannot pad or trim an empty waveform")

    if current_length < target_length:
        repeats = (target_length // current_length) + 1
        waveform = waveform.repeat(1, repeats)
        waveform = waveform[:, :target_length]
    elif current_length > target_length:
        waveform = waveform[:, :target_length]

    return waveform


def compute_eer(scores, labels):
    """Compute Equal Error Rate (EER).

    Args:
        scores: ndarray of shape (N,) - spoof scores (class 1 probabilities)
        labels: ndarray of shape (N,) - 0=bonafide, 1=spoof

    Returns:
        eer: float - Equal Error Rate in percentage
    """
    if len(np.unique(labels)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    eer = np.mean((fpr[np.nanargmin(np.abs(fnr - fpr))], fnr[np.nanargmin(np.abs(fnr - fpr))]))

    return eer * 100


def compute_min_tdcf(scores, labels, p_spoof=0.05, c_miss=1, c_fa=10):
    """Compute a simplified minimum detection cost proxy for a standalone CM system.

    This is not the official ASVspoof tandem DCF — it does not model an external
    ASV system.  It computes a normalized cost for the CM alone:

        t-DCF = (C_miss · FNR · P_bonafide + C_fa · FPR · P_spoof)
                / min(C_miss · P_bonafide, C_fa · P_spoof)

    A value of 0 means perfect detection; 1 means the CM is no better than
    an arbitrarily bad baseline.  Use the official ASVspoof evaluation tooling
    for benchmark reporting.

    Args:
        scores: ndarray of shape (N,) - spoof scores (class 1 probabilities)
        labels: ndarray of shape (N,) - 0=bonafide, 1=spoof
        p_spoof: Prior probability of a spoofing attack (default 0.05, per ASVspoof 2019)
        c_miss: Cost of CM falsely rejecting a bonafide utterance
        c_fa: Cost of CM falsely accepting a spoof

    Returns:
        min_tdcf: float - normalized minimum t-DCF value
    """
    if len(np.unique(labels)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    p_bonafide = 1 - p_spoof

    tdcf = c_miss * fnr * p_bonafide + c_fa * fpr * p_spoof
    norm = min(c_miss * p_bonafide, c_fa * p_spoof)
    min_tdcf = np.min(tdcf) / norm

    return min_tdcf


def keras_lr_decay(step, decay=0.0001):
    """Keras-style learning rate decay.

    Args:
        step: Current step/iteration
        decay: Decay rate

    Returns:
        lr_multiplier: float
    """
    return 1.0 / (1.0 + decay * step)
