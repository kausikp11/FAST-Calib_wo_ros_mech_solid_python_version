from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import open3d as o3d

from data_preprocess import DataPreprocess
from io_utils import listDataPairs, readCameraIntrinsics, saveCalibrationResults
from lidar_detect import LidarDetect
from print_utils import cameraIntrinsicsLines
from qr_detect import QRDetect
from qr_detect_utils import sortPatternCenters
from structs import MAKE_POINTCLOUD, Parameters, Point
from utils import colorPointCloudWithImage, computeRMSE, projectLidarOnImage

LogCallback = Callable[[str], None]


@dataclass
class PipelineConfig:
    cam_intrinsic_file: str
    data_dir: str
    output_dir: str
    marker_size: float = -1.0
    delta_width_qr_center: float = -1.0
    delta_height_qr_center: float = -1.0
    delta_width_circles: float = -1.0
    delta_height_circles: float = -1.0
    min_detected_markers: int = 0
    circle_radius: float = -1.0
    voxel_downsample_size: float = -1.0
    plane_dist_threshold: float = -1.0
    circle_tolerance: float = -1.0
    crop_min_xyz: str = ""
    crop_max_xyz: str = ""
    lidar_center_axis_map: tuple[str, str, str] | None = None


def parseCSV3(value: str):
    if not value:
        return None
    parts = [float(item) for item in value.split(",")]
    return parts if len(parts) == 3 else None


def rigid_transform_svd(source, target):
    src = source.to_xyz()
    dst = target.to_xyz()
    if len(src) == 0 or len(dst) == 0:
        raise ValueError("Source and target point clouds must both be non-empty")
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_centroid - r @ src_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = r
    transform[:3, 3] = t
    return transform


def save_xyz_cloud(path: Path, cloud):
    xyz = cloud.to_xyz().astype(np.float32)
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        f"WIDTH {len(xyz)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(xyz)}",
        "DATA ascii",
    ]
    if len(xyz) == 0:
        body = ""
    else:
        body = "\n".join(f"{pt[0]:.8f} {pt[1]:.8f} {pt[2]:.8f}" for pt in xyz)
    path.write_text("\n".join(header) + "\n" + body + ("\n" if body else ""), encoding="utf-8")


def save_rgb_cloud(path: Path, colored_cloud):
    if not colored_cloud:
        path.write_text(
            "\n".join(
                [
                    "# .PCD v0.7 - Point Cloud Data file format",
                    "VERSION 0.7",
                    "FIELDS x y z rgb",
                    "SIZE 4 4 4 4",
                    "TYPE F F F F",
                    "COUNT 1 1 1 1",
                    "WIDTH 0",
                    "HEIGHT 1",
                    "VIEWPOINT 0 0 0 1 0 0 0",
                    "POINTS 0",
                    "DATA ascii",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return
    xyz = np.array([[p[0], p[1], p[2]] for p in colored_cloud], dtype=np.float64)
    rgb = np.array([[p[3], p[4], p[5]] for p in colored_cloud], dtype=np.float64) / 255.0
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(xyz)
    o3d_cloud.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(str(path), o3d_cloud, write_ascii=True)


class CallbackHandler(logging.Handler):
    def __init__(self, callback: LogCallback | None):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        if self.callback is not None:
            self.callback(self.format(record))


def _emit(message: str, log_callback: LogCallback | None):
    if log_callback is not None:
        log_callback(message)


def build_parameters(config: PipelineConfig):
    cam_params = readCameraIntrinsics(config.cam_intrinsic_file)
    params = Parameters()
    params.camera_intrinsics = cam_params
    if config.marker_size > 0:
        params.marker_size = config.marker_size
    if config.delta_width_qr_center > 0:
        params.delta_width_qr_center = config.delta_width_qr_center
    if config.delta_height_qr_center > 0:
        params.delta_height_qr_center = config.delta_height_qr_center
    if config.delta_width_circles > 0:
        params.delta_width_circles = config.delta_width_circles
    if config.delta_height_circles > 0:
        params.delta_height_circles = config.delta_height_circles
    if config.min_detected_markers > 0:
        params.min_detected_markers = config.min_detected_markers
    if config.circle_radius > 0:
        params.circle_radius = config.circle_radius
    if config.voxel_downsample_size > 0:
        params.voxel_downsample_size = config.voxel_downsample_size
    if config.plane_dist_threshold > 0:
        params.plane_dist_threshold = config.plane_dist_threshold
    if config.circle_tolerance > 0:
        params.circle_tolerance = config.circle_tolerance

    crop_min_xyz = parseCSV3(config.crop_min_xyz)
    crop_max_xyz = parseCSV3(config.crop_max_xyz)
    if crop_min_xyz is not None:
        params.crop_min_xyz = crop_min_xyz
    if crop_max_xyz is not None:
        params.crop_max_xyz = crop_max_xyz
    if config.lidar_center_axis_map is not None:
        params.lidar_center_axis_map = tuple(config.lidar_center_axis_map)
    return cam_params, params


def run_pipeline(config: PipelineConfig, log_callback: LogCallback | None = None):
    logger = logging.getLogger()
    old_handlers = list(logger.handlers)
    old_level = logger.level
    for handler in old_handlers:
        logger.removeHandler(handler)
    callback_handler = CallbackHandler(log_callback)
    callback_handler.setFormatter(logging.Formatter("I%(asctime)s %(filename)s:%(lineno)d] %(message)s", datefmt="%m%d %H:%M:%S"))
    logger.addHandler(callback_handler)
    logger.setLevel(logging.INFO)

    try:
        logging.info("Begin program")
        _emit("=== Program Configuration ===", log_callback)
        _emit(f"Camera intrinsic file: {config.cam_intrinsic_file}", log_callback)
        _emit(f"Data directory: {config.data_dir}", log_callback)
        _emit(f"Output directory: {config.output_dir}", log_callback)
        _emit("=============================", log_callback)

        if not config.cam_intrinsic_file:
            _emit("Warning: Camera intrinsic file not specified", log_callback)
        if not config.output_dir:
            _emit("Warning: Output directory not specified", log_callback)

        cam_params, params = build_parameters(config)
        for line in cameraIntrinsicsLines(cam_params):
            _emit(line, log_callback)

        logging.info("Final Parameters:")
        logging.info(" marker_size = %s", params.marker_size)
        logging.info(" delta_width_qr_center = %s", params.delta_width_qr_center)
        logging.info(" delta_height_qr_center = %s", params.delta_height_qr_center)
        logging.info(" delta_width_circles = %s", params.delta_width_circles)
        logging.info(" delta_height_circles = %s", params.delta_height_circles)
        logging.info(" min_detected_markers = %s", params.min_detected_markers)
        logging.info(" circle_radius = %s", params.circle_radius)
        logging.info(" voxel_downsample_size = %s", params.voxel_downsample_size)
        logging.info(" plane_dist_threshold = %s", params.plane_dist_threshold)
        logging.info(" circle_tolerance = %s", params.circle_tolerance)
        logging.info(" crop_min_xyz = %s", params.crop_min_xyz)
        logging.info(" crop_max_xyz = %s", params.crop_max_xyz)
        logging.info(" lidar_center_axis_map = %s", params.lidar_center_axis_map)

        all_input_data = listDataPairs(config.data_dir)
        _emit(f"Found {len(all_input_data)} .bag and .png file pairs in {config.data_dir}", log_callback)

        dataPreprocessPtr = DataPreprocess(all_input_data)
        if len(dataPreprocessPtr) == 0:
            raise RuntimeError("No valid image/point-cloud pairs were loaded")
        lidarDetectPtr = LidarDetect(params)
        qrDetectPtr = QRDetect(params)

        results = []
        for data_idx in range(min(1, len(dataPreprocessPtr))):
            data_pair_dir = Path(config.output_dir) / f"data_pair_{data_idx}"
            data_pair_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Processing data pair %d / %d", data_idx + 1, len(dataPreprocessPtr))

            lidar_center_cloud = MAKE_POINTCLOUD()
            lidar_center_cloud.reserve(4)
            lidarDetectPtr.detect_mech_lidar(dataPreprocessPtr.cloud_inputs[data_idx], lidar_center_cloud)

            save_xyz_cloud(data_pair_dir / f"edge_cloud_{data_idx}.pcd", lidarDetectPtr.getEdgeCloud())
            logging.info("center_cloud_:")
            save_xyz_cloud(data_pair_dir / f"center_cloud_{data_idx}.pcd", lidar_center_cloud)
            logging.info("center_cloud_:")
            save_xyz_cloud(data_pair_dir / f"plane_cloud_{data_idx}.pcd", lidarDetectPtr.getPlaneCloud())
            logging.info("center_cloud_:")
            save_xyz_cloud(data_pair_dir / f"center_z0_cloud_{data_idx}.pcd", lidarDetectPtr.getCenterZ0Cloud())
            logging.info("center_cloud_:")

            for i, cluster_cloud in enumerate(lidarDetectPtr.getClusterClouds()):
                save_xyz_cloud(data_pair_dir / f"cluster_cloud_{data_idx}_{i}.pcd", cluster_cloud)
            logging.info("Lidar Detection completed ")

            qr_center_cloud = MAKE_POINTCLOUD()
            qrDetectPtr.detect_qr(dataPreprocessPtr.getImage(data_idx), qr_center_cloud)
            cv2.imwrite(str(data_pair_dir / f"qr_img_{data_idx}.png"), qrDetectPtr.imageCopy_)
            logging.info("QR Detection completed ")

            qr_centers = MAKE_POINTCLOUD()
            lidar_centers = MAKE_POINTCLOUD()
            sortPatternCenters(lidar_center_cloud, lidar_centers, "lidar", params)
            sortPatternCenters(qr_center_cloud, qr_centers, "camera")
            save_xyz_cloud(data_pair_dir / f"sort_centers_lidar_{data_idx}.pcd", lidar_centers)
            save_xyz_cloud(data_pair_dir / f"sort_centers_qr_{data_idx}.pcd", qr_centers)

            if len(lidar_centers) != 4 or len(qr_centers) != 4:
                raise RuntimeError(
                    f"Expected 4 sorted centers from lidar and camera, got lidar={len(lidar_centers)} camera={len(qr_centers)}."
                )

            transformation = rigid_transform_svd(lidar_centers, qr_centers)
            _emit(str(transformation[:3, :]), log_callback)
            transformed_cloud = MAKE_POINTCLOUD()
            lidar_xyz = lidar_centers.to_xyz()
            if len(lidar_xyz) > 0:
                transformed_xyz = (transformation[:3, :3] @ lidar_xyz.T).T + transformation[:3, 3]
                for pt in transformed_xyz:
                    transformed_cloud.push_back(Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))
            rmse = computeRMSE(qr_centers, transformed_cloud)
            logging.info("Calibration result RMSE: %s", rmse)

            colored_cloud = []
            cloud_input_downsample = dataPreprocessPtr.getPointCloud(data_idx, params.voxel_downsample_size)
            image_input = dataPreprocessPtr.getImage(data_idx)
            colorPointCloudWithImage(
                cloud_input_downsample,
                transformation,
                cam_params.getCameraMatrix(),
                cam_params.getDistCoeffs(),
                image_input,
                colored_cloud,
            )
            save_rgb_cloud(data_pair_dir / f"colored_cloud_{data_idx}.pcd", colored_cloud)
            projected_overlay = projectLidarOnImage(
                cloud_input_downsample,
                transformation,
                cam_params.getCameraMatrix(),
                cam_params.getDistCoeffs(),
                image_input,
            )
            overlay_path = data_pair_dir / f"lidar_projection_{data_idx}.png"
            cv2.imwrite(str(overlay_path), projected_overlay)
            transform_path = Path(saveCalibrationResults(transformation, str(data_pair_dir)))
            logging.info("Processed data pair %d / %d successfully", data_idx + 1, len(dataPreprocessPtr))
            results.append(
                {
                    "data_pair_dir": data_pair_dir,
                    "overlay_path": overlay_path,
                    "transform_path": transform_path,
                    "rmse": float(rmse),
                }
            )

        _emit("Processing completed!", log_callback)
        return {
            "results": results,
            "params": params,
            "camera_intrinsics": cam_params,
        }
    finally:
        logger.removeHandler(callback_handler)
        for handler in old_handlers:
            logger.addHandler(handler)
        logger.setLevel(old_level)
