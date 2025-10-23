from skimage.measure import marching_cubes
import numpy as np
import open3d as o3d


def pcd_to_mesh(xyz, voxel_size: float = 1.0):
    # Voxelize xyz
    xyz = xyz / voxel_size
    xyz = xyz.astype(np.int32)

    xyz_min = xyz.min(0)
    xyz_max = xyz.max(0) + 1
    z_max = xyz_max[2]

    xyz = xyz[xyz[:, 2] < z_max * 0.8]

    # Make the voxel grid
    voxel_grid = np.zeros(
        (xyz_max[0], xyz_max[1], xyz_max[2]),
        dtype=np.float32,
    )

    # Fill in the voxel grid
    voxel_grid[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = 1.0

    # Marching cubes
    verts, faces, _, _ = marching_cubes(voxel_grid, level=0.5)
    return verts, faces


def write_mesh(verts, faces, path):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(str(path), mesh)
