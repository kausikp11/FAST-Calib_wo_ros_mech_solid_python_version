from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from structs import MAKE_POINTCLOUD, Point


class LidarDetect:
    def __init__(self, params):
        self.filtered_cloud_ = MAKE_POINTCLOUD()
        self.plane_cloud_ = MAKE_POINTCLOUD()
        self.aligned_cloud_ = MAKE_POINTCLOUD()
        self.edge_cloud_ = MAKE_POINTCLOUD()
        self.center_z0_cloud_ = MAKE_POINTCLOUD()
        self.cluster_indices_ = []
        self.cluster_clouds_ = []

        self.x_min_, self.y_min_, self.z_min_ = params.crop_min_xyz
        self.x_max_, self.y_max_, self.z_max_ = params.crop_max_xyz
        self.voxel_down_size_ = params.voxel_downsample_size
        self.circle_radius_ = params.circle_radius
        self.cluster_tolerance_ = params.cluster_tolerance
        self.min_cluster_size_ = params.min_cluster_size
        self.max_cluster_size_ = params.max_cluster_size
        self.delta_width_circles_ = params.delta_width_circles
        self.delta_height_circles_ = params.delta_height_circles
        self.circle_tolerance_ = params.circle_tolerance
        self.plane_dist_threshold_ = params.plane_dist_threshold
        self.target_normal_radius_ = params.target_normal_radius
        self.target_boundary_radius_ = params.target_boundary_radius
        self.target_boundary_angle_thres_ = params.target_boundary_angle_thres

        logging.info(
            "LidarDetect initialized, x_min: %s, x_max: %s, y_min: %s, y_max: %s, z_min: %s, z_max: %s, circle_radius: %s, voxel_down_size: %s",
            self.x_min_,
            self.x_max_,
            self.y_min_,
            self.y_max_,
            self.z_min_,
            self.z_max_,
            self.circle_radius_,
            self.voxel_down_size_,
        )
        logging.info("Plane fitting params, plane_dist_threshold: %s", self.plane_dist_threshold_)
        logging.info(
            "Target params, target_normal_radius: %s, target_boundary_radius: %s, target_boundary_angle_thres: %s",
            self.target_normal_radius_,
            self.target_boundary_radius_,
            self.target_boundary_angle_thres_,
        )

    @staticmethod
    def _cloud_from_mask(cloud, mask):
        result = MAKE_POINTCLOUD()
        for keep, point in zip(mask, cloud.points):
            if keep:
                result.push_back(Point(point.x, point.y, point.z, point.intensity, point.ring))
        return result

    @staticmethod
    def _rotation_to_z(normal):
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = np.cross(normal, z_axis)
        axis_norm = np.linalg.norm(axis)
        dot = np.clip(np.dot(normal, z_axis), -1.0, 1.0)
        if axis_norm < 1e-9:
            return np.eye(3, dtype=np.float64)
        axis = axis / axis_norm
        angle = np.arccos(dot)
        kx, ky, kz = axis
        k = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
        return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)

    def detect_lidar(self, cloud, center_cloud):
        xyz = cloud.to_xyz()
        mask = (
            (xyz[:, 0] >= self.x_min_)
            & (xyz[:, 0] <= self.x_max_)
            & (xyz[:, 1] >= self.y_min_)
            & (xyz[:, 1] <= self.y_max_)
            & (xyz[:, 2] >= self.z_min_)
            & (xyz[:, 2] <= self.z_max_)
        )
        self.filtered_cloud_ = self._cloud_from_mask(cloud, mask)
        logging.info("PassThrough filtered cloud size: %d", len(self.filtered_cloud_))

        if self.voxel_down_size_ > 0 and len(self.filtered_cloud_) > 0:
            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(self.filtered_cloud_.to_xyz())
            down = o3d_cloud.voxel_down_sample(self.voxel_down_size_)
            self.filtered_cloud_ = MAKE_POINTCLOUD()
            for pt in down.points:
                self.filtered_cloud_.push_back(Point(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))
        logging.info("VoxelGrid filtered cloud size: %d", len(self.filtered_cloud_))

        self._segment_plane(self.filtered_cloud_)
        normal = self._plane_normal
        aligned_xyz = (self._R_align @ self.plane_cloud_.to_xyz().T).T
        average_z = float(aligned_xyz[:, 2].mean()) if len(aligned_xyz) else 0.0
        self.aligned_cloud_ = MAKE_POINTCLOUD()
        for pt in aligned_xyz:
            self.aligned_cloud_.push_back(Point(x=float(pt[0]), y=float(pt[1]), z=0.0))

        self.edge_cloud_ = MAKE_POINTCLOUD()
        if len(aligned_xyz) > 0:
            plane_xy = aligned_xyz[:, :2]
            hull = o3d.geometry.PointCloud()
            hull.points = o3d.utility.Vector3dVector(np.column_stack([plane_xy, np.zeros(len(plane_xy))]))
            _, hull_indices = hull.compute_convex_hull()
            hull_index_set = set(hull_indices)
            for idx in hull_index_set:
                p = self.aligned_cloud_[idx]
                self.edge_cloud_.push_back(Point(x=p.x, y=p.y, z=p.z))
        logging.info("Edge cloud size: %d", len(self.edge_cloud_))

        self.cluster_clouds_ = []
        self.cluster_indices_ = []
        if len(self.edge_cloud_) > 0:
            labels = DBSCAN(eps=self.cluster_tolerance_, min_samples=self.min_cluster_size_).fit_predict(
                self.edge_cloud_.to_xyz()[:, :2]
            )
            for label in sorted(set(labels)):
                if label < 0:
                    continue
                indices = np.where(labels == label)[0].tolist()
                if not (self.min_cluster_size_ <= len(indices) <= self.max_cluster_size_):
                    continue
                self.cluster_indices_.append(indices)
                cluster = MAKE_POINTCLOUD()
                for idx in indices:
                    cluster.push_back(self.edge_cloud_[idx])
                self.cluster_clouds_.append(cluster)
        logging.info("Number of edge clusters: %d", len(self.cluster_indices_))

        self.center_z0_cloud_ = MAKE_POINTCLOUD()
        r_inv = np.linalg.inv(self._R_align)
        for cluster in self.cluster_clouds_:
            xy = cluster.to_xyz()[:, :2]
            if len(xy) < 3:
                continue
            try:
                model, inliers = ransac(xy, CircleModel, min_samples=3, residual_threshold=0.01, max_trials=5000)
            except ValueError:
                continue
            if model is None or inliers is None or np.count_nonzero(inliers) == 0:
                continue
            xc, yc, _ = model.params
            error = float(np.mean(np.abs(np.linalg.norm(xy[inliers] - np.array([xc, yc]), axis=1) - self.circle_radius_)))
            if error < 0.025:
                center_point = Point(x=float(xc), y=float(yc), z=0.0)
                self.center_z0_cloud_.push_back(center_point)
                aligned_point = np.array([xc, yc, average_z], dtype=np.float64)
                original_point = r_inv @ aligned_point
                center_cloud.push_back(Point(x=float(original_point[0]), y=float(original_point[1]), z=float(original_point[2])))
        center_cloud.width = 1
        center_cloud.height = len(center_cloud)

    def _segment_plane(self, cloud):
        self.plane_cloud_ = MAKE_POINTCLOUD()
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(cloud.to_xyz())
        plane_model, inliers = o3d_cloud.segment_plane(distance_threshold=self.plane_dist_threshold_, ransac_n=3, num_iterations=5000)
        a, b, c, _ = plane_model
        for idx in inliers:
            p = cloud[idx]
            self.plane_cloud_.push_back(Point(x=p.x, y=p.y, z=p.z, intensity=p.intensity, ring=p.ring))
        logging.info("Plane cloud size: %d", len(self.plane_cloud_))
        normal = np.array([a, b, c], dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        self._plane_normal = normal
        self._R_align = self._rotation_to_z(normal)

    def detect_mech_lidar(self, cloud, center_cloud):
        self.cluster_clouds_ = []
        xyz = cloud.to_xyz()
        mask = (
            (xyz[:, 0] >= self.x_min_)
            & (xyz[:, 0] <= self.x_max_)
            & (xyz[:, 1] >= self.y_min_)
            & (xyz[:, 1] <= self.y_max_)
            & (xyz[:, 2] >= self.z_min_)
            & (xyz[:, 2] <= self.z_max_)
        )
        self.filtered_cloud_ = self._cloud_from_mask(cloud, mask)
        logging.info("[mech] Depth filtered cloud size: %d", len(self.filtered_cloud_))

        self._segment_plane(self.filtered_cloud_)
        plane_xyz = self.plane_cloud_.to_xyz()
        plane_o3d = o3d.geometry.PointCloud()
        plane_o3d.points = o3d.utility.Vector3dVector(plane_xyz)
        plane_model, _ = plane_o3d.segment_plane(distance_threshold=self.plane_dist_threshold_, ransac_n=3, num_iterations=5000)
        c = plane_model
        normal = np.array(c[:3], dtype=np.float64)
        norm_n = np.linalg.norm(normal)
        normal = normal / norm_n

        ring2indices = defaultdict(list)
        for i, pt in enumerate(self.filtered_cloud_):
            ring2indices[pt.ring].append(i)

        self.edge_cloud_ = MAKE_POINTCLOUD()
        neighbor_gap_threshold = 0.10
        min_points_per_ring = 2
        for idx_vec in ring2indices.values():
            if len(idx_vec) < min_points_per_ring:
                continue
            for k in range(1, len(idx_vec) - 1):
                p_prev = self.filtered_cloud_[idx_vec[k - 1]]
                p_cur = self.filtered_cloud_[idx_vec[k]]
                p_next = self.filtered_cloud_[idx_vec[k + 1]]
                dist_plane = abs(c[0] * p_cur.x + c[1] * p_cur.y + c[2] * p_cur.z + c[3]) / norm_n
                if dist_plane >= 0.03:
                    continue
                dist_prev = float(np.linalg.norm(np.array([p_cur.x - p_prev.x, p_cur.y - p_prev.y, p_cur.z - p_prev.z])))
                dist_next = float(np.linalg.norm(np.array([p_cur.x - p_next.x, p_cur.y - p_next.y, p_cur.z - p_next.z])))
                if dist_prev > neighbor_gap_threshold or dist_next > neighbor_gap_threshold:
                    self.edge_cloud_.push_back(Point(x=p_cur.x, y=p_cur.y, z=p_cur.z, ring=p_cur.ring))
        logging.info("[mech] Extracted %d edge points (neighbor distance).", len(self.edge_cloud_))

        self.aligned_cloud_ = MAKE_POINTCLOUD()
        r_align = self._rotation_to_z(normal)
        average_z = 0.0
        for pt in self.edge_cloud_:
            aligned_point = r_align @ np.array([pt.x, pt.y, pt.z], dtype=np.float64)
            self.aligned_cloud_.push_back(Point(x=float(aligned_point[0]), y=float(aligned_point[1]), z=0.0, ring=pt.ring))
            average_z += float(aligned_point[2])
        if len(self.edge_cloud_) > 0:
            average_z /= len(self.edge_cloud_)

        xy_cloud = self.aligned_cloud_.copy()
        logging.info("[mech] Start circle detection, initial cloud size = %d", len(xy_cloud))
        self.center_z0_cloud_ = MAKE_POINTCLOUD()

        while len(xy_cloud) > 3:
            logging.info("[mech] RANSAC on cloud of size %d", len(xy_cloud))
            xy = xy_cloud.to_xyz()[:, :2]
            center_xy, radius, inliers = self._fit_fixed_radius_circle(
                xy,
                self.circle_radius_,
                self.plane_dist_threshold_,
                5000,
            )

            if center_xy is None or inliers is None or np.count_nonzero(inliers) == 0:
                logging.info("[mech] No more circles can be found, stop.")
                break

            xc, yc = center_xy
            inlier_count = int(np.count_nonzero(inliers))
            logging.info("[mech] circle r = %s, inliers = %d", radius, inlier_count)
            if radius < self.circle_radius_ - self.circle_tolerance_ or radius > self.circle_radius_ + self.circle_tolerance_:
                logging.info("[mech] coeff")
                break
            if inlier_count < 20:
                logging.info("[mech] Circle inliers too few (%d), stop.", inlier_count)
                break

            self.center_z0_cloud_.push_back(Point(x=float(xc), y=float(yc), z=0.0))
            remaining = MAKE_POINTCLOUD()
            for keep, point in zip(~inliers, xy_cloud.points):
                if keep:
                    remaining.push_back(point)
            xy_cloud = remaining
        logging.info("[mech] Detected %d raw circle candidates before geometry filter.", len(self.center_z0_cloud_))

        r_inv = np.linalg.inv(r_align)
        center_cloud.clear()
        for cpt in self.center_z0_cloud_:
            aligned_point = np.array([cpt.x, cpt.y, cpt.z + average_z], dtype=np.float64)
            original_point = r_inv @ aligned_point
            center_cloud.push_back(Point(x=float(original_point[0]), y=float(original_point[1]), z=float(original_point[2])))
        center_cloud.width = 1
        center_cloud.height = len(center_cloud)

    @staticmethod
    def _circle_centers_from_pair(p1, p2, radius):
        delta = p2 - p1
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9 or distance > 2.0 * radius:
            return []

        midpoint = 0.5 * (p1 + p2)
        height_sq = radius * radius - (distance * 0.5) ** 2
        if height_sq < 0.0:
            return []

        height = float(np.sqrt(max(height_sq, 0.0)))
        perp = np.array([-delta[1], delta[0]], dtype=np.float64) / distance
        return [midpoint + height * perp, midpoint - height * perp]

    @staticmethod
    def _refine_circle_center(points, initial_center, radius):
        center = initial_center.astype(np.float64).copy()
        for _ in range(20):
            diffs = points - center
            dists = np.linalg.norm(diffs, axis=1)
            valid = dists > 1e-9
            if not np.any(valid):
                break
            unit = diffs[valid] / dists[valid, None]
            residuals = dists[valid] - radius
            jtj = unit.T @ unit
            jtr = unit.T @ residuals
            try:
                step = np.linalg.solve(jtj, jtr)
            except np.linalg.LinAlgError:
                break
            center += step
            if np.linalg.norm(step) < 1e-6:
                break
        return center

    def _fit_fixed_radius_circle(self, points_xy, radius, residual_threshold, max_trials):
        if len(points_xy) < 3:
            return None, None, None

        rng = np.random.default_rng(0)
        best_center = None
        best_inliers = None
        best_count = 0
        best_score = float("inf")

        for _ in range(max_trials):
            sample_indices = rng.choice(len(points_xy), size=2, replace=False)
            p1 = points_xy[sample_indices[0]]
            p2 = points_xy[sample_indices[1]]
            for center in self._circle_centers_from_pair(p1, p2, radius):
                residuals = np.abs(np.linalg.norm(points_xy - center, axis=1) - radius)
                inliers = residuals <= residual_threshold
                count = int(np.count_nonzero(inliers))
                if count == 0:
                    continue
                score = float(np.mean(residuals[inliers]))
                if count > best_count or (count == best_count and score < best_score):
                    best_center = center
                    best_inliers = inliers
                    best_count = count
                    best_score = score

        if best_center is None or best_inliers is None:
            return None, None, None

        refined_center = self._refine_circle_center(points_xy[best_inliers], best_center, radius)
        refined_residuals = np.abs(np.linalg.norm(points_xy - refined_center, axis=1) - radius)
        refined_inliers = refined_residuals <= residual_threshold
        if np.count_nonzero(refined_inliers) < best_count:
            refined_inliers = best_inliers
            refined_center = best_center

        estimated_radius = float(np.mean(np.linalg.norm(points_xy[refined_inliers] - refined_center, axis=1)))
        return refined_center, estimated_radius, refined_inliers

    def getFilteredCloud(self):
        return self.filtered_cloud_

    def getPlaneCloud(self):
        return self.plane_cloud_

    def getAlignedCloud(self):
        return self.aligned_cloud_

    def getEdgeCloud(self):
        return self.edge_cloud_

    def getCenterZ0Cloud(self):
        return self.center_z0_cloud_

    def getClusterClouds(self):
        return self.cluster_clouds_
