from __future__ import annotations

import cv2
import open3d as o3d

from io_utils import loadPointCloudFromNPZ, loadPointCloudFromTXT
from structs import MAKE_POINTCLOUD, Point


class DataPreprocess:
    def __init__(self, all_input_data):
        self.input_data_infos = []
        self.cloud_inputs = []
        self.img_inputs = []
        for input_data in all_input_data:
            img_input = cv2.imread(input_data.img_file, cv2.IMREAD_UNCHANGED)
            if img_input is None:
                print(f"Error: Could not load image: {input_data.img_file}")
                continue

            cloud_input = MAKE_POINTCLOUD()
            for pcd_file in input_data.pcd_files:
                suffix = pcd_file.lower()
                if suffix.endswith(".npz"):
                    temp_cloud = loadPointCloudFromNPZ(pcd_file)
                else:
                    temp_cloud = loadPointCloudFromTXT(pcd_file)
                if len(temp_cloud) == 0:
                    print(f"Warning: Loaded empty point cloud from: {pcd_file}")
                    continue
                cloud_input.extend(temp_cloud)

            if len(cloud_input) == 0:
                print(f"Error: No valid point cloud data loaded for image: {input_data.img_file}")
                continue

            self.input_data_infos.append(input_data)
            self.cloud_inputs.append(cloud_input)
            self.img_inputs.append(img_input)
            print(
                f"Loaded image: {input_data.img_file} with size {img_input.shape[1]}x{img_input.shape[0]} "
                f"and point cloud with {len(cloud_input)} points."
            )

    def __len__(self):
        return len(self.img_inputs)

    def getImage(self, index: int):
        if 0 <= index < len(self.img_inputs):
            return self.img_inputs[index]
        raise IndexError(f"Pair index {index} is out of range for {len(self.img_inputs)} loaded pair(s)")

    def getPointCloud(self, index: int, voxel_downsample_size: float):
        if 0 <= index < len(self.cloud_inputs):
            cloud = self.cloud_inputs[index]
            if voxel_downsample_size > 0:
                o3d_cloud = o3d.geometry.PointCloud()
                o3d_cloud.points = o3d.utility.Vector3dVector(cloud.to_xyz())
                downsampled = o3d_cloud.voxel_down_sample(voxel_downsample_size)
                downsample_cloud = MAKE_POINTCLOUD()
                for point in downsampled.points:
                    downsample_cloud.push_back(Point(x=float(point[0]), y=float(point[1]), z=float(point[2])))
                return downsample_cloud
            return cloud
        raise IndexError(f"Pair index {index} is out of range for {len(self.cloud_inputs)} loaded pair(s)")
