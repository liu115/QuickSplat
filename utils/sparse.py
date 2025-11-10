from typing import List, Tuple, Optional, Literal

import numpy as np
import torch
import pytorch3d.ops
import pytorch3d.loss
import MinkowskiEngine as ME
import open3d as o3d


def xyz_list_to_bxyz(
    xyz_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[int]]:
    """
    Args:
        xyz_list: list of torch.Tensor, each of shape (N, 3). Can have different lengths
    """
    lengths = [len(xyz) for xyz in xyz_list]
    bxyz = []
    for i, xyz in enumerate(xyz_list):
        bxyz.append(
            torch.cat([
                torch.full((len(xyz), 1), i, dtype=torch.int32, device=xyz.device),
                xyz.int(),
            ], dim=1)
        )

    return torch.cat(bxyz, dim=0), lengths


def list_tensor_to_sparse(
    xyz_list: List[torch.Tensor],
    feat_list: List[torch.Tensor],
) -> Tuple[ME.SparseTensor, List[int]]:
    """
    Args:
        xyz_list: list of torch.Tensor, each of shape (N, 3). Can have different lengths
        feat_list: list of torch.Tensor, each of shape (N, C). Can have different lengths
    """

    bxyz, lengths = xyz_list_to_bxyz(xyz_list)
    features = torch.cat(feat_list, dim=0)

    sparse_tensor = ME.SparseTensor(
        features=features,
        coordinates=bxyz,
    )
    return sparse_tensor, lengths


def bxyz_to_list_xyz(
    bxyz: torch.Tensor,
    batch_size: int,
    features: Optional[torch.Tensor] = None,
    lengths: Optional[List[int]] = None,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    assert bxyz.ndim == 2, "bxyz must have shape (N, 4)"
    assert bxyz.shape[1] == 4, "bxyz must have shape (N, 4)"
    if features is not None:
        assert features.ndim == 2, "features must have shape (N, C)"
        assert features.shape[0] == bxyz.shape[0], "features must have shape (N, C)"

    xyz_list = []
    feature_list = []

    if lengths is not None:
        acc_length = [0]
        for l in lengths:
            acc_length.append(acc_length[-1] + l)

        for i in range(len(lengths)):
            start = acc_length[i]
            end = acc_length[i + 1]
            xyz_list.append(bxyz[start:end, 1:])
            if features is not None:
                feature_list.append(features[start:end])

    else:
        for i in range(batch_size):
            mask = bxyz[:, 0] == i
            xyz_list.append(bxyz[mask, 1:])
            if features is not None:
                feature_list.append(features[mask])

    if features is not None:
        return xyz_list, feature_list
    else:
        return xyz_list


def sparse_tensor_to_lists(
    sparse_tensor: ME.SparseTensor,
    batch_size: int,
    lengths: Optional[List[int]] = None,
):
    return bxyz_to_list_xyz(
        sparse_tensor.C.clone(),
        batch_size,
        sparse_tensor.F,
        lengths,
    )


def find_non_exist_points(
    xyz_source: torch.Tensor,
    xyz_target: torch.Tensor,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the points in xyz_target that are not in xyz_source
    Args:
        xyz_source: torch.Tensor of shape (N, 3)
        xyz_target: torch.Tensor of shape (M, 3)
        eps: float, small number to avoid numerical error
    Returns:
        xyz_target[indices]: torch.Tensor of shape (K, 3), K <= M
        indices: torch.Tensor of shape (K,)
    """
    _dist, indices, _ = pytorch3d.ops.ball_query(
        xyz_target.unsqueeze(0).float(),
        xyz_source.unsqueeze(0).float(),
        K=1,
        radius=1 - eps,
        return_nn=False,
    )
    indices = indices.flatten()
    valid_mask = (indices < 0)

    non_exist_points = xyz_target[valid_mask]
    non_exist_indices = torch.nonzero(valid_mask).flatten()
    return non_exist_points, non_exist_indices


# def find_unique_points(
#     xyz: torch.Tensor,
#     eps: float = 1e-4,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Find the points in xyz that are unique
#     """
#     _dist, indices, _ = pytorch3d.ops.ball_query(
#         xyz.unsqueeze(0).float(),
#         xyz.unsqueeze(0).float(),
#         K=2,
#         radius=eps,
#         return_nn=False,
#     )

#     indices = indices.squeeze(0)
#     valid_mask = (indices < 0).any(dim=1)
#     unique_points = xyz[valid_mask]
#     unique_indices = torch.nonzero(valid_mask).flatten()
#     return unique_points, unique_indices


def get_neighboring_offsets(device, include_self=False):
    if include_self:
        return torch.tensor([
            0, 0, 0,
            1, 0, 0,
            -1, 0, 0,
            0, 1, 0,
            0, -1, 0,
            0, 0, 1,
            0, 0, -1,
        ], dtype=torch.int32, device=device).reshape(-1, 3)
    return torch.tensor([
        1, 0, 0,
        -1, 0, 0,
        0, 1, 0,
        0, -1, 0,
        0, 0, 1,
        0, 0, -1,
    ], dtype=torch.int32, device=device).reshape(-1, 3)


def chamfer_dist(
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
    norm: int = 2,
    single_directional: bool = False,
    **kwargs,
):
    assert xyz1.ndim == 2, "xyz1 must have shape (N, 3)"
    assert xyz2.ndim == 2, "xyz2 must have shape (M, 3)"
    assert xyz1.shape[1] == 3, "xyz1 must have shape (N, 3)"
    assert xyz2.shape[1] == 3, "xyz2 must have shape (M, 3)"

    loss, _ = pytorch3d.loss.chamfer_distance(
        xyz1.unsqueeze(0).float(),
        xyz2.unsqueeze(0).float(),
        norm=norm,
        single_directional=single_directional,
        **kwargs,
    )
    return loss


def chamfer_dist_with_crop(
    xyz_pred: torch.Tensor,
    xyz_gt: torch.Tensor,
    boundary_ratio: float = 1.2,
    norm: int = 2,
    direction: Literal["bidirection", "pred_to_gt", "gt_to_pred"] = "bidirection",
    **kwargs,
):
    assert direction in ["bidirection", "pred_to_gt", "gt_to_pred"], f"Invalid direction: {direction}"
    # Keep only the points in xyz_gt * boundary_ratio
    xyz_min = xyz_gt.min(dim=0).values
    xyz_max = xyz_gt.max(dim=0).values

    size = xyz_max - xyz_min
    xyz_min = xyz_min - size * (boundary_ratio - 1) / 2
    xyz_max = xyz_max + size * (boundary_ratio - 1) / 2

    valid_mask = (xyz_pred[:, 0] >= xyz_min[0]) & (xyz_pred[:, 0] <= xyz_max[0])
    valid_mask &= (xyz_pred[:, 1] >= xyz_min[1]) & (xyz_pred[:, 1] <= xyz_max[1])
    valid_mask &= (xyz_pred[:, 2] >= xyz_min[2]) & (xyz_pred[:, 2] <= xyz_max[2])

    xyz_pred_cropped = xyz_pred[valid_mask]

    single_directional = direction != "bidirection"
    if direction == "pred_to_gt" or direction == "bidirection":
        loss, _ = pytorch3d.loss.chamfer_distance(
            xyz_pred_cropped.unsqueeze(0).float(),
            xyz_gt.unsqueeze(0).float(),
            norm=norm,
            single_directional=single_directional,
            **kwargs,
        )
    else:   # direction == "gt_to_pred"
        loss, _ = pytorch3d.loss.chamfer_distance(
            xyz_gt.unsqueeze(0).float(),
            xyz_pred_cropped.unsqueeze(0).float(),
            norm=norm,
            single_directional=single_directional,
            **kwargs,
        )
    return loss


def chamfer_dist_with_margin(
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
    norm: int = 2,
    margin: float = 0.1,
):
    (dist1, dist2), _ = pytorch3d.loss.chamfer_distance(
        xyz1.unsqueeze(0).float(),
        xyz2.unsqueeze(0).float(),
        norm=norm,
        batch_reduction=None,
        point_reduction=None,
    )
    # dist1 and dist2 are of shape (N,P2) and (N,P1)
    dist1 = dist1.squeeze(0)    # (P2,)
    dist2 = dist2.squeeze(0)    # (P1,)
    dist = torch.cat([dist1, dist2], dim=0)
    loss = torch.clamp(margin - dist, min=0)
    return loss.mean()


def compute_sparse_iou(
    xyz_pred: torch.Tensor,
    xyz_gt: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Args:
        xyz_pred: (N, 3) int tensor
        xyz_gt: (M, 3) int tensor
    Returns:
        precision: float, precision
        recall: float, recall
        iou: float, intersection over union
        f1: float, f1 score
    """

    xyz_pred = xyz_pred.int()
    xyz_gt = xyz_gt.int()

    pred_set = set([tuple(x) for x in xyz_pred.cpu().numpy()])
    gt_set = set([tuple(x) for x in xyz_gt.cpu().numpy()])
    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set.difference(gt_set))
    fn = len(gt_set.difference(pred_set))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    # print(f"precision: {precision}, recall: {recall}, iou: {iou}, f1: {f1}")
    return precision, recall, iou, f1


def points_in_mask(xyz: torch.Tensor, known_mask: torch.Tensor):
    valid_mask = torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    valid_mask &= (x >= 0) & (x < known_mask.shape[0])
    valid_mask &= (y >= 0) & (y < known_mask.shape[1])
    valid_mask &= (z >= 0) & (z < known_mask.shape[2])
    x = x.clamp(0, known_mask.shape[0] - 1).int()
    y = y.clamp(0, known_mask.shape[1] - 1).int()
    z = z.clamp(0, known_mask.shape[2] - 1).int()
    # mask = known_mask[x, y, z]
    valid_mask &= (known_mask[x, y, z] > 0)
    return xyz[valid_mask, :]


def chamfer_dist_with_mask(
    xyz_pred: torch.Tensor,
    xyz_gt: torch.Tensor,
    known_mask: torch.Tensor,
    norm: int = 2,
):
    """Compute chamfer distance the points in the known_mask

    Args:
        xyz_pred (torch.Tensor): (N, 3) int tensor
        xyz_gt (torch.Tensor): (M, 3) int tensor
        known_mask (torch.Tensor): (H, W, D) bool tensor
        norm (int, optional)
    """
    assert xyz_pred.ndim == 2, "xyz1 must have shape (N, 3)"
    assert xyz_gt.ndim == 2, "xyz2 must have shape (M, 3)"
    assert xyz_pred.shape[1] == 3, "xyz1 must have shape (N, 3)"
    assert xyz_gt.shape[1] == 3, "xyz2 must have shape (M, 3)"
    assert known_mask.ndim == 3, "mask must have shape (H, W, D)"

    xyz_pred_masked = points_in_mask(xyz_pred, known_mask)
    xyz_gt_masked = points_in_mask(xyz_gt, known_mask)
    # print(f"Pred: Before: {xyz_pred.shape[0]}, After: {xyz_pred_masked.shape[0]}")
    # print(f"GT: Before: {xyz_gt.shape[0]}, After: {xyz_gt_masked.shape[0]}")

    loss, _ = pytorch3d.loss.chamfer_distance(
        xyz_pred_masked.unsqueeze(0).float(),
        xyz_gt_masked.unsqueeze(0).float(),
        norm=norm,
    )
    return loss


def compute_sparse_iou_with_mask(
    xyz_pred: torch.Tensor,
    xyz_gt: torch.Tensor,
    known_mask: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Args:
        xyz_pred: (N, 3) int tensor
        xyz_gt: (M, 3) int tensor
        known_mask: (H, W, D) bool tensor
    Returns:
        precision: float, precision
        recall: float, recall
        iou: float, intersection over union
        f1: float, f1 score
    """

    xyz_pred = points_in_mask(xyz_pred, known_mask)
    xyz_gt = points_in_mask(xyz_gt, known_mask)

    pred_set = set([tuple(x) for x in xyz_pred.cpu().numpy()])
    gt_set = set([tuple(x) for x in xyz_gt.cpu().numpy()])
    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set.difference(gt_set))
    fn = len(gt_set.difference(pred_set))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    # print(f"precision: {precision}, recall: {recall}, iou: {iou}, f1: {f1}")
    return precision, recall, iou, f1


def chamfer_dist_mesh_with_crop(
    verts_input: torch.Tensor,
    faces_input: torch.Tensor,
    verts_gt: torch.Tensor,
    faces_gt: torch.Tensor,
    boundary_ratio: float = 1.2,
    norm: int = 2,
    direction: Literal["bidirection", "pred_to_gt", "gt_to_pred"] = "bidirection",
    num_sampled_points: int = 200_000,
    **kwargs,
):
    """
    Compute chamfer distance between two meshes by uniformly a fix number of points on the meshes
    """
    mesh_input = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_input.cpu().numpy()),
        triangles=o3d.utility.Vector3iVector(faces_input.cpu().numpy()),
    )
    mesh_gt = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_gt.cpu().numpy()),
        triangles=o3d.utility.Vector3iVector(faces_gt.cpu().numpy()),
    )

    pcd_input = mesh_input.sample_points_uniformly(number_of_points=num_sampled_points)
    pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=num_sampled_points)

    xyz_input = torch.from_numpy(np.asarray(pcd_input.points)).to(verts_input.device)
    xyz_gt = torch.from_numpy(np.asarray(pcd_gt.points)).to(verts_gt.device)

    return chamfer_dist_with_crop(
        xyz_input,
        xyz_gt,
        boundary_ratio,
        norm,
        direction,
        **kwargs,
    )
