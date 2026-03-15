from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from structs import CameraIntrinsics, InputDataInfo, MAKE_POINTCLOUD, Point


def listDataPairs(data_dir: str) -> list[InputDataInfo]:
    all_input_datas: list[InputDataInfo] = []
    data_path = Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        print(f"Error: Data directory does not exist or is not a directory: {data_dir}")
        return all_input_datas

    png_files = sorted([str(p) for p in data_path.iterdir() if p.is_file() and p.suffix == ".png"])
    for img_file in png_files:
        png_path = Path(img_file)
        base_name = png_path.stem
        expected_bag = png_path.parent / f"{base_name}.bag"
        if not expected_bag.exists():
            print(f"Warning: No matching BAG file found for {img_file}")

        pcd_dir = png_path.parent / f"{base_name}_pcd"
        if not pcd_dir.exists():
            print(f"Warning: No matching PCD directory found for {img_file}")
            raise RuntimeError(f"Missing PCD directory for {img_file}")

        pcd_files = sorted(
            str(p)
            for p in pcd_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".txt", ".npz"}
        )
        if not pcd_files:
            print(f"Warning: No .txt or .npz files found in PCD directory: {pcd_dir}")
            raise RuntimeError(f"No supported point-cloud files in {pcd_dir}")

        all_input_datas.append(
            InputDataInfo(
                base_path=str(png_path.parent),
                bag_file=str(expected_bag),
                img_file=img_file,
                pcd_dir=str(pcd_dir),
                pcd_files=pcd_files,
            )
        )
    return all_input_datas


def readCameraIntrinsics(filepath: str) -> CameraIntrinsics:
    with open(filepath, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return CameraIntrinsics(
        cam_model=config["cam_model"],
        cam_width=int(config["cam_width"]),
        cam_height=int(config["cam_height"]),
        fx=float(config["cam_fx"]),
        fy=float(config["cam_fy"]),
        cx=float(config["cam_cx"]),
        cy=float(config["cam_cy"]),
        k1=float(config["cam_d0"]),
        k2=float(config["cam_d1"]),
        p1=float(config["cam_d2"]),
        p2=float(config["cam_d3"]),
    )


def loadPointCloudFromTXT(filepath: str):
    cloud = MAKE_POINTCLOUD()
    if not os.path.exists(filepath):
        print(f"Error: Could not open file {filepath}")
        return cloud
    with open(filepath, "r", encoding="utf-8") as handle:
        for line in handle:
            values = line.strip().split()
            if len(values) < 5:
                break
            x, y, z, intensity, ring = map(float, values[:5])
            cloud.push_back(Point(x=x, y=y, z=z, intensity=float(intensity), ring=int(ring)))
    return cloud


def loadPointCloudFromNPZ(filepath: str):
    cloud = MAKE_POINTCLOUD()
    if not os.path.exists(filepath):
        print(f"Error: Could not open file {filepath}")
        return cloud

    data = np.load(filepath)
    required_keys = {"xyz_m", "intensity", "ring", "distance_m"}
    missing_keys = required_keys.difference(data.files)
    if missing_keys:
        raise KeyError(f"NPZ file {filepath} is missing keys: {sorted(missing_keys)}")

    xyz = np.asarray(data["xyz_m"], dtype=np.float64)
    intensity = np.asarray(data["intensity"], dtype=np.float64).reshape(-1)
    ring = np.asarray(data["ring"], dtype=np.float64).reshape(-1)
    distance = np.asarray(data["distance_m"], dtype=np.float64).reshape(-1)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"NPZ file {filepath} has invalid xyz_m shape: {xyz.shape}")

    point_count = xyz.shape[0]
    if not (len(intensity) == len(ring) == len(distance) == point_count):
        raise ValueError(
            f"NPZ file {filepath} has inconsistent lengths: "
            f"xyz={point_count}, intensity={len(intensity)}, ring={len(ring)}, distance={len(distance)}"
        )

    valid = distance > 0.1
    xyz = xyz[valid]
    intensity = intensity[valid]
    ring = ring[valid]

    for idx in range(len(xyz)):
        cloud.push_back(
            Point(
                x=float(xyz[idx, 0]),
                y=float(xyz[idx, 1]),
                z=float(xyz[idx, 2]),
                intensity=float(intensity[idx]),
                ring=int(ring[idx]),
            )
        )
    return cloud


def saveCalibrationResults(transformation: np.ndarray, dirPath: str) -> str:
    output_path = Path(dirPath) / "output.txt"
    np.savetxt(output_path, transformation, fmt="%.8f")
    return str(output_path)
