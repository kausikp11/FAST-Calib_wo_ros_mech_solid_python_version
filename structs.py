from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field

import numpy as np

TARGET_NUM_CIRCLES = 4
DEBUG = 1
GEOMETRY_TOLERANCE = 0.08


@dataclass
class CameraIntrinsics:
    cam_model: str = ""
    cam_width: int = 0
    cam_height: int = 0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def getCameraMatrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def getDistCoeffs(self) -> np.ndarray:
        return np.array([[self.k1, self.k2, self.p1, self.p2, 0.0]], dtype=np.float64)


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    intensity: float = 0.0
    ring: int = 0


class PointCloud:
    def __init__(self, points: list[Point] | None = None) -> None:
        self.points: list[Point] = list(points) if points is not None else []
        self.width = 1
        self.height = len(self.points)

    def push_back(self, point: Point) -> None:
        self.points.append(point)
        self.height = len(self.points)

    def reserve(self, _: int) -> None:
        return None

    def clear(self) -> None:
        self.points.clear()
        self.height = 0

    def resize(self, size: int) -> None:
        current = len(self.points)
        if size < current:
            del self.points[size:]
        else:
            self.points.extend(Point() for _ in range(size - current))
        self.height = len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int) -> Point:
        return self.points[index]

    def __setitem__(self, index: int, value: Point) -> None:
        self.points[index] = value

    def extend(self, other: "PointCloud") -> None:
        self.points.extend(other.points)
        self.height = len(self.points)

    def copy(self) -> "PointCloud":
        return PointCloud([Point(p.x, p.y, p.z, p.intensity, p.ring) for p in self.points])

    def to_xyz(self) -> np.ndarray:
        if not self.points:
            return np.empty((0, 3), dtype=np.float64)
        return np.array([[p.x, p.y, p.z] for p in self.points], dtype=np.float64)


@dataclass
class InputDataInfo:
    base_path: str = ""
    bag_file: str = ""
    img_file: str = ""
    pcd_dir: str = ""
    pcd_files: list[str] = field(default_factory=list)


@dataclass
class Parameters:
    camera_intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    marker_size: float = 0.2
    delta_width_qr_center: float = 0.55
    delta_height_qr_center: float = 0.55
    delta_width_circles: float = 0.5
    delta_height_circles: float = 0.4
    min_detected_markers: int = 3
    circle_radius: float = 0.12
    crop_min_xyz: list[float] = field(default_factory=lambda: [-1.6, 1.0, -1.0])
    crop_max_xyz: list[float] = field(default_factory=lambda: [0.5, 3.5, 1.0])
    voxel_downsample_size: float = 0.005
    plane_dist_threshold: float = 0.05
    circle_tolerance: float = 0.005
    target_normal_radius: float = 0.1
    target_boundary_radius: float = 0.1
    target_boundary_angle_thres: float = math.pi / 2
    cluster_tolerance: float = 0.1
    min_cluster_size: int = 5
    max_cluster_size: int = 650
    lidar_center_axis_map: tuple[str, str, str] = ("x", "-z", "y")

    @staticmethod
    def _axis_token_to_vector(token: str) -> np.ndarray:
        axis = token.strip().lower()
        basis = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        }
        if axis not in basis:
            raise ValueError(f"Unsupported lidar_center_axis_map token: {token}")
        return basis[axis]

    def get_lidar_center_axis_matrix(self) -> np.ndarray:
        matrix = np.vstack([self._axis_token_to_vector(token) for token in self.lidar_center_axis_map])
        if matrix.shape != (3, 3) or not np.isclose(abs(np.linalg.det(matrix)), 1.0):
            raise ValueError(f"Invalid lidar_center_axis_map: {self.lidar_center_axis_map}")
        return matrix

    def get_lidar_center_axis_inverse(self) -> np.ndarray:
        return np.linalg.inv(self.get_lidar_center_axis_matrix())


def comb(n: int, k: int, groups: list[list[int]]) -> None:
    if DEBUG:
        logging.info("%d centers found. Iterating over %d possible sets of candidates", n, math.comb(n, k))
    groups.extend([list(group) for group in itertools.combinations(range(n), k)])


class Square:
    def __init__(self, candidates: list[Point], width: float, height: float) -> None:
        self._candidates = candidates
        self._target_width = width
        self._target_height = height
        self._target_diagonal = math.sqrt(width * width + height * height)
        self._center = Point()
        if candidates:
            inv = 1.0 / len(candidates)
            self._center.x = sum(p.x for p in candidates) * inv
            self._center.y = sum(p.y for p in candidates) * inv
            self._center.z = sum(p.z for p in candidates) * inv

    @staticmethod
    def distance(a: Point, b: Point) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    def is_valid(self) -> bool:
        if len(self._candidates) != 4:
            return False

        from qr_detect_utils import sortPatternCenters

        half_diag = self._target_diagonal * 0.5
        for p in self._candidates:
            d = self.distance(self._center, p)
            if abs(d - half_diag) / half_diag > GEOMETRY_TOLERANCE * 2.0:
                return False

        candidates_cloud = PointCloud(self._candidates)
        sorted_centers = PointCloud()
        sortPatternCenters(candidates_cloud, sorted_centers, "camera")

        s01 = self.distance(sorted_centers[0], sorted_centers[1])
        s12 = self.distance(sorted_centers[1], sorted_centers[2])
        s23 = self.distance(sorted_centers[2], sorted_centers[3])
        s30 = self.distance(sorted_centers[3], sorted_centers[0])

        pattern1 = (
            abs(s01 - self._target_width) / self._target_width < GEOMETRY_TOLERANCE
            and abs(s12 - self._target_height) / self._target_height < GEOMETRY_TOLERANCE
            and abs(s23 - self._target_width) / self._target_width < GEOMETRY_TOLERANCE
            and abs(s30 - self._target_height) / self._target_height < GEOMETRY_TOLERANCE
        )
        pattern2 = (
            abs(s01 - self._target_height) / self._target_height < GEOMETRY_TOLERANCE
            and abs(s12 - self._target_width) / self._target_width < GEOMETRY_TOLERANCE
            and abs(s23 - self._target_height) / self._target_height < GEOMETRY_TOLERANCE
            and abs(s30 - self._target_width) / self._target_width < GEOMETRY_TOLERANCE
        )
        if not pattern1 and not pattern2:
            return False

        perimeter = s01 + s12 + s23 + s30
        ideal = 2.0 * (self._target_width + self._target_height)
        return abs(perimeter - ideal) / ideal <= GEOMETRY_TOLERANCE


def MAKE_POINTCLOUD() -> PointCloud:
    return PointCloud()
