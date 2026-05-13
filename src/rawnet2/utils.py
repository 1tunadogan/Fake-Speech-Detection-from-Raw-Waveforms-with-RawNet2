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
    torch.backends.cudnn.benchmark = False


def pad_or_trim(waveform, target_length):
    """Pad short waveforms by tiling, crop long ones.

    Args:
        waveform: Tensor of shape (channels, time)
        target_length: Target number of samples

    Returns:
        Tensor of shape (channels, target_length)
    """
    current_length = waveform.shape[1]

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
    # Edge case: all labels are the same class
    if len(np.unique(labels)) < 2:
        return 0.0

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find the threshold where |FAR - FRR| is minimized
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    return eer * 100


def compute_min_tdcf(scores, labels, p_target=0.05, c_miss=1, c_fa=10):
    """Compute minimum normalized tandem Detection Cost Function (t-DCF).

    Args:
        scores: ndarray of shape (N,) - spoof scores
        labels: ndarray of shape (N,) - 0=bonafide, 1=spoof
        p_target: Prior probability of target class (bonafide)
        c_miss: Cost of miss
        c_fa: Cost of false alarm

    Returns:
        min_tdcf: float - minimum t-DCF value
    """
    if len(np.unique(labels)) < 2:
        return 0.0

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    tdcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    min_tdcf = np.min(tdcf)

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
