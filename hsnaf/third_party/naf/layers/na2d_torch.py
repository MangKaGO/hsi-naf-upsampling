"""Pure-PyTorch 2D neighborhood attention (fallback when NATTEN is not installed).

Written for this project, not part of the original NAF code base.

It follows NATTEN's ``na2d`` semantics: with dilation ``d``, the query grid is
split into ``d`` interleaved sub-grids per axis, and each query attends to a
``k x k`` window of keys from its own sub-grid. Near the borders the window is
shifted (not truncated) so that it always contains ``k`` keys.

It is much slower and more memory hungry than NATTEN's fused kernels, but it
runs anywhere (CPU, Windows, macOS) and is fine for inference on small images.
"""

import torch


def _neighborhood_indices(length: int, kernel_size: int, dilation: int, device) -> torch.Tensor:
    """For each position along one axis, the ``kernel_size`` key indices it attends to.

    Returns a ``(length, kernel_size)`` long tensor.
    """
    pos = torch.arange(length, device=device)
    group = pos % dilation                      # which dilated sub-grid
    pos_in_group = pos // dilation              # position inside that sub-grid
    group_len = (length - group + dilation - 1) // dilation
    if int(group_len.min()) < kernel_size:
        raise ValueError(
            f"kernel_size={kernel_size} is larger than a dilated sub-grid "
            f"(axis length {length}, dilation {dilation})."
        )
    start = (pos_in_group - kernel_size // 2).clamp(min=0)
    start = torch.minimum(start, group_len - kernel_size)
    offsets = torch.arange(kernel_size, device=device)
    return group[:, None] + dilation * (start[:, None] + offsets[None, :])


def na2d(q, k, v, kernel_size, dilation=(1, 1), scale=None):
    """Neighborhood attention.

    Args:
        q, k: ``(batch, H, W, heads, head_dim)``
        v: ``(batch, H, W, heads, value_dim)``
        kernel_size: int or ``(kh, kw)``
        dilation: int or ``(dh, dw)``
        scale: attention scale, defaults to ``head_dim ** -0.5``

    Returns:
        ``(batch, H, W, heads, value_dim)``
    """
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
    _, H, W, _, head_dim = q.shape
    scale = head_dim ** -0.5 if scale is None else scale

    idx_h = _neighborhood_indices(H, kh, dh, q.device)  # (H, kh)
    idx_w = _neighborhood_indices(W, kw, dw, q.device)  # (W, kw)

    # Attention logits for every neighbor: (batch, H, W, heads, kh * kw)
    logits = torch.stack(
        [
            (q * k[:, idx_h[:, a]][:, :, idx_w[:, b]]).sum(-1)
            for a in range(kh)
            for b in range(kw)
        ],
        dim=-1,
    ) * scale
    weights = logits.softmax(dim=-1)

    out = torch.zeros(*q.shape[:-1], v.shape[-1], dtype=v.dtype, device=v.device)
    for n, (a, b) in enumerate((a, b) for a in range(kh) for b in range(kw)):
        out += weights[..., n, None] * v[:, idx_h[:, a]][:, :, idx_w[:, b]]
    return out
