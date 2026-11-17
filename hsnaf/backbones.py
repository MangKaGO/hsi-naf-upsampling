"""Frozen pretrained backbones: DOFA (feature extractor) and SpectralEarth's spectral adapter."""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block

from .third_party.dofa import Dynamic_MLP_OFA
from .third_party.spectral_earth import SpectralAdapter

DOFA_INPUT_SIZE = 224


class DOFAEncoder(nn.Module):
    """DOFA ViT-L/16 encoder returning the patch tokens of the last block as a feature map.

    DOFA's wavelength-conditioned patch embedding lets the same ViT ingest images
    with any number of spectral bands. For a 224x224 input, the output is a
    ``(batch, 1024, 14, 14)`` feature map (classification head and final norm are
    not used).
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4.0, wv_planes=128):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.embed_dim = embed_dim

        self.patch_embed = Dynamic_MLP_OFA(wv_planes=wv_planes, kernel_size=patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Fixed sin-cos position embedding, loaded from the checkpoint
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size**2 + 1, embed_dim), requires_grad=False)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)]
        )

    def forward(self, x, wavelengths):
        """
        Args:
            x: ``(batch, bands, 224, 224)``
            wavelengths: sequence of ``bands`` wavelengths
        """
        waves = torch.as_tensor(wavelengths, dtype=torch.float, device=x.device)
        x, _ = self.patch_embed(x, waves)
        x = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)

        tokens = x[:, 1:, :]  # drop the class token
        return tokens.transpose(1, 2).reshape(x.shape[0], self.embed_dim, self.grid_size, self.grid_size)


def _freeze(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    return module


def load_dofa(checkpoint_path, device="cpu"):
    """Builds DOFA ViT-L/16 and loads the ``DOFA_ViT_large_e100.pth`` weights (frozen, eval mode)."""
    model = DOFAEncoder()
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"DOFA checkpoint is missing weights: {missing}")
    return _freeze(model).to(device)


def load_spectral_adapter(checkpoint_path, device="cpu"):
    """Loads SpectralEarth's spectral adapter (any number of bands -> 128 channels, frozen, eval mode).

    The weights are taken from a full SpectralEarth checkpoint (e.g. ``spec_ViTb_mae.pth``);
    only the ``spectral_adapter.*`` entries are used.
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    prefix = "spectral_adapter."
    adapter_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    adapter = SpectralAdapter()
    adapter.load_state_dict(adapter_state)
    return _freeze(adapter).to(device)


def resize_for_dofa(x, mode="nearest"):
    """Brings an image to DOFA's 224x224 input size (upsampling if smaller, top-left crop if larger)."""
    if x.shape[-1] < DOFA_INPUT_SIZE or x.shape[-2] < DOFA_INPUT_SIZE:
        x = F.interpolate(x, size=(DOFA_INPUT_SIZE, DOFA_INPUT_SIZE), mode=mode)
    return x[..., :DOFA_INPUT_SIZE, :DOFA_INPUT_SIZE]


@torch.no_grad()
def dofa_features(dofa, x, wavelengths, resize_mode="nearest"):
    """DOFA features of an image of any size: ``(batch, bands, H, W)`` -> ``(batch, 1024, 14, 14)``."""
    return dofa(resize_for_dofa(x, resize_mode), wavelengths)
