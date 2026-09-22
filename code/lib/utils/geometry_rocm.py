"""
ROCm-compatible replacements for the NVIDIA Kaolin operations used by X-Avatar.

Designed for:
    PyTorch ROCm
    PyTorch3D
    AMD Instinct MI250X

This module preserves X-Avatar's geometry operations while removing its
runtime dependency on NVIDIA Kaolin.
"""

import torch
from pytorch3d.ops import knn_points
from pytorch3d import _C


def index_vertices_by_faces(vertices, faces):
    """
    Replacement for kaolin.ops.mesh.index_vertices_by_faces.

    vertices: [B, V, 3]
    faces:    [F, 3]

    returns:  [B, F, 3, 3]
    """
    return vertices[:, faces.long(), :]


def sided_distance(points_a, points_b):
    """
    Replacement for kaolin.metrics.pointcloud.sided_distance.

    Returns squared nearest-neighbour distance and nearest-point index.
    """
    result = knn_points(points_a, points_b, K=1)

    dist2 = result.dists[..., 0]
    idx = result.idx[..., 0]

    return dist2, idx


def point_to_mesh_distance(
    points,
    face_vertices,
    min_triangle_area=5e-3
):
    """
    Replacement for the X-Avatar use of
    kaolin.metrics.trianglemesh.point_to_mesh_distance.

    Returns:
        squared distance
        closest face index
        distance type placeholder

    X-Avatar uses the closest-face index.
    """

    if points.shape[0] != 1 or face_vertices.shape[0] != 1:
        raise NotImplementedError(
            "ROCm point_to_mesh_distance currently supports batch size 1."
        )

    points_flat = points[0].contiguous()
    tris_flat = face_vertices[0].contiguous()

    points_first_idx = torch.tensor(
        [0],
        dtype=torch.int64,
        device=points.device
    )

    tris_first_idx = torch.tensor(
        [0],
        dtype=torch.int64,
        device=points.device
    )

    dist2, idx = _C.point_face_dist_forward(
        points_flat,
        points_first_idx,
        tris_flat,
        tris_first_idx,
        points_flat.shape[0],
        float(min_triangle_area),
    )

    # X-Avatar expects three outputs from Kaolin.
    distance_type = torch.zeros_like(idx)

    return (
        dist2.unsqueeze(0),
        idx.unsqueeze(0),
        distance_type.unsqueeze(0)
    )


def check_sign(vertices, faces, points, eps=1e-8):
    """
    ROCm/PyTorch inside-outside test replacing kaolin.ops.mesh.check_sign.

    Uses ray/triangle intersection parity:
        odd number of intersections  -> inside
        even number                 -> outside

    vertices: [B, V, 3]
    faces:    [F, 3]
    points:   [B, P, 3]

    returns:
        bool tensor [B, P]
    """

    if vertices.ndim != 3 or points.ndim != 3:
        raise ValueError("vertices and points must be batched tensors.")

    batch_size = vertices.shape[0]

    # X-Avatar normally supplies common topology [F,3].
    if faces.ndim == 2:
        faces_batch = faces.unsqueeze(0).expand(batch_size, -1, -1)
    elif faces.ndim == 3:
        faces_batch = faces
    else:
        raise ValueError("faces must have shape [F,3] or [B,F,3].")

    direction = torch.tensor(
        [1.0, 0.371, 0.529],
        dtype=vertices.dtype,
        device=vertices.device
    )
    direction = direction / torch.linalg.norm(direction)

    batch_results = []

    for b in range(batch_size):

        tris = vertices[b][faces_batch[b].long()]

        v0 = tris[:, 0]
        v1 = tris[:, 1]
        v2 = tris[:, 2]

        edge1 = v1 - v0
        edge2 = v2 - v0

        point_results = []

        for p in points[b]:

            d = direction.expand_as(edge2)

            h = torch.cross(d, edge2, dim=1)
            a = torch.sum(edge1 * h, dim=1)

            valid = torch.abs(a) > eps

            inv_a = torch.zeros_like(a)
            inv_a[valid] = 1.0 / a[valid]

            s = p.unsqueeze(0) - v0

            u = inv_a * torch.sum(s * h, dim=1)

            q = torch.cross(s, edge1, dim=1)

            v = inv_a * torch.sum(d * q, dim=1)
            t = inv_a * torch.sum(edge2 * q, dim=1)

            hit = (
                valid
                & (u >= 0.0)
                & (u <= 1.0)
                & (v >= 0.0)
                & ((u + v) <= 1.0)
                & (t > eps)
            )

            point_results.append((hit.sum() % 2) == 1)

        batch_results.append(torch.stack(point_results))

    return torch.stack(batch_results)
