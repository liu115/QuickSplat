import numpy as np
import open3d as o3d


def draw_cameras(vis, world_to_camera, intrinsic, color):
    outs = []
    for i in range(world_to_camera.shape[0]):
        camera = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=int(intrinsic[i, 0, 2] * 2),
            view_height_px=int(intrinsic[i, 1, 2] * 2),
            intrinsic=intrinsic[i],
            extrinsic=world_to_camera[i],
            scale=0.4,
        )
        camera.paint_uniform_color(color)
        camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        camera_to_world = np.linalg.inv(world_to_camera[i])
        camera_axis.transform(camera_to_world)
        # vis.add_geometry(camera_axis)
        vis.add_geometry(camera)
        outs.append(camera)
        outs.append(camera_axis)
    return outs


def draw_bbox(vis, bbox, color):
    print(bbox[0, 0])
    print(bbox[0, 1])
    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox[0, 0], bbox[0, 1])
    bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    bbox.paint_uniform_color(color)
    vis.add_geometry(bbox)
    return bbox


class FakeVis:
    def create_window(self):
        pass
    def add_geometry(self, geom):
        pass
    def run(self):
        pass
    def destroy_window(self):
        pass
