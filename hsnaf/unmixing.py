"""Hyperspectral unmixing from foundation-model features.

Linear mixing model: each pixel ``y = E a``, with ``E`` the ``(bands, c)`` endmember
matrix and ``a >= 0, sum(a) = 1`` the abundances. A small network maps the
(upsampled) features to abundances, and a non-negative linear decoder
(whose weights are the endmembers) reconstructs the image. Training is
unsupervised: the loss only compares the reconstruction to the input image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import spectral_angle


def sivm(Y, c):
    """Simplex Volume Maximization endmember extraction (adapted from UnDIP).

    Args:
        Y: image, ``(bands, H, W)``, ``(1, bands, H, W)`` or ``(bands, pixels)``
        c: number of endmembers

    Returns:
        ``(bands, c)`` endmembers (projected on the first ``c`` principal components).
    """
    Y = Y.reshape(Y.shape[-3] if Y.dim() > 2 else Y.shape[0], -1)
    U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    Yp = U[:, :c] @ (torch.diag(S[:c]) @ Vh[:c])

    # First endmember: the pixel farthest from the origin, then iteratively the
    # pixel farthest from the subspace spanned by the selected ones.
    selected = torch.argmax((Y**2).sum(0)).reshape(1)
    for _ in range(1, c):
        E = Yp[:, selected]
        projector = E @ torch.linalg.pinv(E.T @ E) @ E.T
        dist = ((Yp - projector @ Yp) ** 2).sum(0)
        dist[selected] = -torch.inf
        selected = torch.cat([selected, torch.argmax(dist).reshape(1)])
    return Yp[:, selected]


class UnmixingNet(nn.Module):
    """Features ``(1, D, h, w)`` -> abundances ``(1, c, H, W)`` -> reconstruction ``(1, bands, H, W)``.

    Args:
        feature_dim: D, number of feature channels (1024 for DOFA)
        n_bands: number of spectral bands of the image
        n_endmembers: c
        feature_size: h (= w) of the input features
        image_size: H (= W) of the abundance maps
    """

    def __init__(self, feature_dim, n_bands, n_endmembers, feature_size, image_size):
        super().__init__()
        self.image_size = image_size

        # Learned spatial upsampling of every feature channel (h*w -> H*W)
        self.upsample = nn.Linear(feature_size**2, image_size**2)
        self.to_abundances = nn.Sequential(
            nn.Conv2d(feature_dim, n_endmembers, kernel_size=1, bias=False),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(n_endmembers),
            nn.Dropout(0.2),
        )
        # Decoder weights = endmembers (kept non-negative, see clamp_endmembers)
        self.decoder = nn.Conv2d(n_endmembers, n_bands, kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    @property
    def endmembers(self):
        """``(bands, c)``"""
        return self.decoder.weight[:, :, 0, 0]

    @torch.no_grad()
    def init_endmembers(self, E):
        self.decoder.weight[:, :, 0, 0] = E

    @torch.no_grad()
    def clamp_endmembers(self):
        self.decoder.weight.clamp_(min=0)

    def abundances(self, features):
        feats = features.flatten(2)                               # (1, D, h*w)
        feats = self.upsample(feats).unflatten(2, (self.image_size, self.image_size))
        feats = (feats - feats.mean()) / (feats.std() + 1e-8)
        return F.softmax(self.to_abundances(feats), dim=1)       # sum-to-one

    def forward(self, features):
        A = self.abundances(features)
        Y = F.relu(self.decoder(A))
        return self.endmembers, A, Y


def unmixing_loss(Y, Y_hat, A_hat, E_hat, w_sad=1.0, w_mse=0.09, w_sparsity=0.6, w_e=3e-5):
    """Reconstruction (SAD + normalized MSE) + abundance sparsity + endmember regularization."""
    sad = spectral_angle(Y, Y_hat, dim=1).mean()
    mse = ((Y - Y_hat) ** 2).sum() / (Y**2).sum()
    sparsity = torch.sqrt(A_hat).mean()
    # L1 difference between consecutive endmember columns
    e_reg = (E_hat[:, 1:] - E_hat[:, :-1]).abs().sum()
    return w_sad * sad + w_mse * mse + w_sparsity * sparsity + w_e * e_reg


def train_unmixing(features, Y, n_endmembers, epochs=200, lr=2e-3, weight_decay=0.18, progress=True):
    """Fits an :class:`UnmixingNet` on precomputed (frozen) features of image ``Y``.

    Args:
        features: ``(1, D, h, w)``
        Y: ``(1, bands, H, W)``, the image to reconstruct (H must equal W)

    Returns:
        model, E_hat ``(bands, c)``, A_hat ``(1, c, H, W)``
    """
    from tqdm import trange

    model = UnmixingNet(
        feature_dim=features.shape[1], n_bands=Y.shape[1], n_endmembers=n_endmembers,
        feature_size=features.shape[-1], image_size=Y.shape[-1],
    ).to(Y.device)
    model.init_endmembers(sivm(Y / Y.max(), n_endmembers))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    for _ in trange(epochs, desc="unmixing", disable=not progress):
        optimizer.zero_grad()
        E_hat, A_hat, Y_hat = model(features)
        loss = unmixing_loss(Y, Y_hat, A_hat, E_hat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=1)
        optimizer.step()
        model.clamp_endmembers()

    model.eval()
    with torch.no_grad():
        E_hat, A_hat, _ = model(features)
    return model, E_hat, A_hat
