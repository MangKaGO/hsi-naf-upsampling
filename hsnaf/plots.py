"""Figures for unmixing results and feature maps."""

import matplotlib.pyplot as plt
import torch

COLORS = ["mediumpurple", "indianred", "cornflowerblue", "goldenrod", "mediumseagreen", "lightpink"]


def plot_endmembers(E_gt, results, title="Endmembers"):
    """Ground-truth vs estimated endmembers (each scaled to max 1), with per-endmember SAD."""
    E_gt, E_hat = E_gt.cpu(), results["E_hat"]
    c = E_gt.shape[1]
    ncols = (c + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 5), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= c:
            ax.axis("off")
            continue
        ax.plot(E_gt[:, i] / E_gt[:, i].max(), "r", lw=1, label="ground truth")
        ax.plot(E_hat[:, i] / E_hat[:, i].max(), "k", lw=1, label="estimated")
        ax.set_title(f"SAD = {results['sad_per_endmember'][i]:.3f}", color="white",
                     backgroundcolor=COLORS[i % len(COLORS)], fontsize=10)
        ax.set_xlabel("band")
    axes.flat[0].legend()
    fig.suptitle(f"{title} — mean SAD = {results['sad']:.3f}")
    fig.tight_layout()
    return fig


def plot_abundances(A_gt, results, title="Abundances"):
    """Estimated (top) vs ground-truth (bottom) abundance maps, with per-endmember NMSE."""
    A_gt, A_hat = A_gt.cpu().squeeze(0), results["A_hat"]
    c = A_gt.shape[0]
    fig, axes = plt.subplots(2, c, figsize=(2.6 * c, 5.2), squeeze=False)
    for i in range(c):
        axes[0, i].imshow(A_hat[i], vmin=0, vmax=1)
        axes[0, i].set_title(f"NMSE = {results['nmse_per_endmember'][i]:.3f}", color="white",
                             backgroundcolor=COLORS[i % len(COLORS)], fontsize=10)
        axes[1, i].imshow(A_gt[i], vmin=0, vmax=1)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("estimated")
    axes[1, 0].set_ylabel("ground truth")
    fig.suptitle(f"{title} — NMSE = {results['nmse']:.3f}")
    fig.tight_layout()
    return fig


def pca_rgb(x):
    """Projects a ``(C, H, W)`` feature map / image on its first 3 principal components, as an RGB image."""
    C, H, W = x.shape
    X = x.detach().float().cpu().reshape(C, -1).T
    X = X - X.mean(0)
    _, _, V = torch.pca_lowrank(X, q=3, center=False)
    rgb = X @ V[:, :3]
    rgb = (rgb - rgb.min(0).values) / (rgb.max(0).values - rgb.min(0).values + 1e-8)
    return rgb.reshape(H, W, 3).numpy()
