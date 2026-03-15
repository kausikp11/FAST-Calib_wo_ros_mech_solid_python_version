from __future__ import annotations

import math

import numpy as np

from structs import MAKE_POINTCLOUD, Parameters, Point


def sortPatternCenters(in_pc, out_v, axis_mode: str = "camera", params: Parameters | None = None):
    if len(in_pc) != 4:
        print(f"[sortPatternCenters] Number of {axis_mode} center points to be sorted is not 4.")
        return

    work_pc = MAKE_POINTCLOUD()
    if axis_mode == "lidar":
        axis_params = params if params is not None else Parameters()
        lidar_to_camera = axis_params.get_lidar_center_axis_matrix()
        for p in in_pc:
            transformed = lidar_to_camera @ np.array([p.x, p.y, p.z], dtype=np.float64)
            work_pc.push_back(Point(x=float(transformed[0]), y=float(transformed[1]), z=float(transformed[2])))
    else:
        work_pc = in_pc.copy()

    xyz = work_pc.to_xyz()
    centroid = xyz.mean(axis=0)
    proj_points = []
    for index, p in enumerate(work_pc.points):
        rel_vec = np.array([p.x - centroid[0], p.y - centroid[1], p.z - centroid[2]], dtype=np.float64)
        proj_points.append((math.atan2(rel_vec[1], rel_vec[0]), index))
    proj_points.sort(key=lambda item: item[0])

    out_v.resize(4)
    for i in range(4):
        out_v[i] = work_pc[proj_points[i][1]]

    p0 = out_v.points[0]
    p1 = out_v.points[1]
    p2 = out_v.points[2]
    v01 = np.array([p1.x - p0.x, p1.y - p0.y, 0.0], dtype=np.float64)
    v12 = np.array([p2.x - p1.x, p2.y - p1.y, 0.0], dtype=np.float64)
    if np.cross(v01, v12)[2] > 0:
        out_v.points[1], out_v.points[3] = out_v.points[3], out_v.points[1]

    if axis_mode == "lidar":
        camera_to_lidar = (params if params is not None else Parameters()).get_lidar_center_axis_inverse()
        for point in out_v.points:
            original = camera_to_lidar @ np.array([point.x, point.y, point.z], dtype=np.float64)
            point.x = float(original[0])
            point.y = float(original[1])
            point.z = float(original[2])
