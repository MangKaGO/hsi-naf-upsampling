"""Unmixing metrics: spectral angle distance (SAD) and normalized MSE, with endmember matching."""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def spectral_angle(x, y, dim=0, eps=1e-7):
    """Angle (radians) between ``x`` and ``y`` along ``dim``."""
    cos = (F.normalize(x, dim=dim) * F.normalize(y, dim=dim)).sum(dim)
    return torch.acos(cos.clamp(-1 + eps, 1 - eps))


def match_endmembers(E_gt, E_hat):
    """Permutation of the estimated endmembers that best matches the ground truth.

    Solves the assignment problem on the spectral angle between the columns of
    ``E_gt`` and ``E_hat`` (both ``(bands, c)``). Returns ``perm`` such that
    ``E_hat[:, perm]`` is aligned with ``E_gt``.
    """
    cost = spectral_angle(E_gt[:, :, None], E_hat[:, None, :], dim=0)  # (c, c)
    _, perm = linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(perm)


@torch.no_grad()
def evaluate_unmixing(E_hat, A_hat, E_gt, A_gt):
    """Matches endmembers to the ground truth and computes the metrics.

    Args:
        E_hat, E_gt: endmembers, ``(bands, c)``
        A_hat, A_gt: abundances, ``(c, H, W)`` or ``(1, c, H, W)``

    Returns a dict with the aligned ``E_hat`` / ``A_hat`` and
        sad_per_endmember, sad (mean), nmse_per_endmember, nmse (global).
    Abundances are compared after scaling each map set to unit Frobenius norm.
    """
    E_hat, E_gt = E_hat.detach().cpu(), E_gt.detach().cpu()
    A_hat, A_gt = A_hat.detach().cpu().squeeze(0), A_gt.detach().cpu().squeeze(0)

    perm = match_endmembers(E_gt, E_hat)
    E_hat, A_hat = E_hat[:, perm], A_hat[perm]

    sad = spectral_angle(E_gt, E_hat, dim=0)

    A_hat_n, A_gt_n = A_hat / A_hat.norm(), A_gt / A_gt.norm()
    sq_err = ((A_gt_n - A_hat_n) ** 2).sum(dim=(1, 2))
    nmse = sq_err / (A_gt_n**2).sum(dim=(1, 2))

    return {
        "E_hat": E_hat,
        "A_hat": A_hat,
        "sad_per_endmember": sad.tolist(),
        "sad": sad.mean().item(),
        "nmse_per_endmember": nmse.tolist(),
        "nmse": (sq_err.sum() / (A_gt_n**2).sum()).item(),
    }
