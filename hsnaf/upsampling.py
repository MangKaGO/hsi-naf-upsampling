"""NAF feature upsampling for hyperspectral images.

Pipeline (see README):
    HSI --(SpectralEarth spectral adapter)--> 128-channel guidance image --> NAF image encoder
    HSI --(frozen DOFA)--> low-resolution features (14x14x1024)        --> NAF attention values
NAF then predicts features at any higher resolution with a cross-scale
neighborhood attention guided by the image.
"""

import torch
import torch.nn.functional as F

from .backbones import DOFA_INPUT_SIZE, dofa_features
from .third_party.naf import NAF

ADAPTER_CHANNELS = 128  # output channels of SpectralEarth's spectral adapter


def build_naf(kernel_size=9, checkpoint_path=None, device="cpu"):
    """NAF taking the 128-channel spectral-adapter output as guidance image.

    ``kernel_size`` is the neighborhood size of the attention. It has no learned
    parameters, so a checkpoint can be used with any kernel size, as long as it
    fits in the low-resolution grid (7x7 -> 14x14 training requires ``kernel_size <= 7``).
    """
    naf = NAF(in_channels=ADAPTER_CHANNELS, kernel_size=kernel_size)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        naf.load_state_dict(state["model_state_dict"])
    return naf.to(device)


@torch.no_grad()
def upsample_features(naf, adapter, dofa, hsi, wavelengths, output_size=None):
    """DOFA features of ``hsi`` upsampled by NAF to ``output_size`` (default: the image size).

    Returns ``(batch, 1024, *output_size)``.
    """
    output_size = output_size or tuple(hsi.shape[-2:])
    guidance = adapter(hsi)
    low_res = dofa_features(dofa, hsi, wavelengths)
    return naf(guidance, low_res, output_size=output_size)


# --------------------------------------------------------------------------- #
# Training pairs: NAF is trained to predict higher-resolution DOFA features.  #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def make_pair_7to14(dofa, hsi, wavelengths):
    """Self-supervised pair: DOFA features (14x14) are the target, their bilinear 7x7 downsampling the input."""
    hsi_224 = F.interpolate(hsi, size=(DOFA_INPUT_SIZE, DOFA_INPUT_SIZE), mode="bilinear", align_corners=False)
    target = dofa(hsi_224, wavelengths)
    low_res = F.interpolate(target, size=(7, 7), mode="bilinear", align_corners=False)
    return low_res, target


@torch.no_grad()
def make_pair_14to28(dofa, hsi, wavelengths):
    """Pair built from quadrants: the target is a 28x28 mosaic of the DOFA features of the
    4 image quadrants (each upsampled to 224x224), the input the DOFA features of the whole image."""
    def features(x):
        x = F.interpolate(x, size=(DOFA_INPUT_SIZE, DOFA_INPUT_SIZE), mode="bilinear", align_corners=False)
        return dofa(x, wavelengths)

    h, w = hsi.shape[-2] // 2, hsi.shape[-1] // 2
    quadrants = [[features(hsi[..., i * h:(i + 1) * h, j * w:(j + 1) * w]) for j in range(2)] for i in range(2)]
    target = torch.cat([torch.cat(row, dim=-1) for row in quadrants], dim=-2)
    return features(hsi), target


TRAINING_MODES = {
    # mode: (pair builder, output size, default kernel size)
    "7to14": (make_pair_7to14, (14, 14), 7),
    "14to28": (make_pair_14to28, (28, 28), 9),
}
