from typing import Optional

import torch
import MinkowskiEngine as ME

def dense_to_sparse(
    x: torch.Tensor,
    stride: int,
    device: torch.device,
    coordinate_manager: Optional[ME.CoordinateManager] = None,
) -> ME.SparseTensor:
    """Convert torch 3D tensor to sparse tensor

    Args:
        x (torch.Tensor): shape (B, C, D, H, W)
        stride (int): How many downsampled times

    Returns:
        ME.SparseTensor that has the size (B * D * H * W, C)
    """
    assert x.ndim == 5
    with torch.no_grad():
        coords = torch.meshgrid(
            torch.arange(x.shape[0], device=device, dtype=torch.int32),
            torch.arange(x.shape[2], device=device, dtype=torch.int32),
            torch.arange(x.shape[3], device=device, dtype=torch.int32),
            torch.arange(x.shape[4], device=device, dtype=torch.int32),
            indexing="ij",
        )
        coords = torch.stack([c.flatten() for c in coords], dim=-1)
    coords[:, 1:] *= stride
    feats = x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[1])
    out = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        tensor_stride=stride,
        device=x.device,
        coordinate_manager=coordinate_manager,
    )
    return out
