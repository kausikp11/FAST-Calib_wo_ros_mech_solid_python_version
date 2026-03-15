from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from structs import MAKE_POINTCLOUD, Point, Square, comb


class QRDetect:
    def __init__(self, params):
        self.marker_size_ = params.marker_size
        self.delta_width_qr_center_ = params.delta_width_qr_center
        self.delta_height_qr_center_ = params.delta_height_qr_center
        self.delta_width_circles_ = params.delta_width_circles
        self.delta_height_circles_ = params.delta_height_circles
        self.min_detected_markers_ = params.min_detected_markers
        cam_params = params.camera_intrinsics
        self.cameraMatrix_ = np.array(
            [[cam_params.fx, 0.0, cam_params.cx], [0.0, cam_params.fy, cam_params.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.distCoeffs_ = np.array([[cam_params.k1, cam_params.k2, cam_params.p1, cam_params.p2, 0.0]], dtype=np.float32)
        self.dictionary_ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.imageCopy_ = None
        logging.info(
            "QR detect initialized with parameters:\nmarker_size: %s\ndelta_width_qr_center: %s\ndelta_height_qr_center: %s\ndelta_width_circles: %s\ndelta_height_circles: %s\nmin_detected_markers: %s",
            self.marker_size_,
            self.delta_width_qr_center_,
            self.delta_height_qr_center_,
            self.delta_width_circles_,
            self.delta_height_circles_,
            self.min_detected_markers_,
        )

    def projectPointDist(self, pt_cv, intrinsics, distCoeffs):
        input_points = np.array([[pt_cv]], dtype=np.float32)
        projected_points, _ = cv2.projectPoints(
            input_points,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            intrinsics,
            distCoeffs,
        )
        return projected_points[0, 0]

    def detect_qr(self, image, centers_cloud):
        self.imageCopy_ = image.copy()
        boardCorners = []
        boardCircleCenters = []
        width = self.delta_width_qr_center_
        height = self.delta_height_qr_center_
        circle_width = self.delta_width_circles_ / 2.0
        circle_height = self.delta_height_circles_ / 2.0
        for i in range(4):
            x_qr_center = -1 if (i % 3) == 0 else 1
            y_qr_center = 1 if i < 2 else -1
            x_center = x_qr_center * width
            y_center = y_qr_center * height
            boardCircleCenters.append(np.array([x_qr_center * circle_width, y_qr_center * circle_height, 0.0], dtype=np.float32))
            marker_corners = []
            for j in range(4):
                x_qr = -1 if (j % 3) == 0 else 1
                y_qr = 1 if j < 2 else -1
                marker_corners.append(
                    [x_center + x_qr * self.marker_size_ / 2.0, y_center + y_qr * self.marker_size_ / 2.0, 0.0]
                )
            boardCorners.append(np.array(marker_corners, dtype=np.float32))

        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(self.dictionary_, parameters)
        corners, ids, _ = detector.detectMarkers(image)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(self.imageCopy_, corners, ids)

        ids_list = [] if ids is None else ids.flatten().tolist()
        if self.min_detected_markers_ <= len(ids_list) <= 4:
            marker_object_points = np.array(
                [
                    [-self.marker_size_ / 2.0, self.marker_size_ / 2.0, 0.0],
                    [self.marker_size_ / 2.0, self.marker_size_ / 2.0, 0.0],
                    [self.marker_size_ / 2.0, -self.marker_size_ / 2.0, 0.0],
                    [-self.marker_size_ / 2.0, -self.marker_size_ / 2.0, 0.0],
                ],
                dtype=np.float32,
            )

            marker_layout = []
            for marker_index, marker_corners in enumerate(corners):
                center = marker_corners.reshape(4, 2).mean(axis=0)
                marker_layout.append((float(center[0]), float(center[1]), marker_index))
            marker_layout.sort(key=lambda item: (item[1], item[0]))
            top_row = sorted(marker_layout[:2], key=lambda item: item[0])
            bottom_row = sorted(marker_layout[2:], key=lambda item: item[0])
            ordered_marker_indices = [top_row[0][2], top_row[1][2], bottom_row[1][2], bottom_row[0][2]]

            marker_rvecs = []
            marker_tvecs = []
            for marker_index in ordered_marker_indices:
                marker_corners = corners[marker_index]
                ok, mrvec, mtvec = cv2.solvePnP(marker_object_points, marker_corners.reshape(-1, 2), self.cameraMatrix_, self.distCoeffs_)
                if not ok:
                    continue
                cv2.drawFrameAxes(self.imageCopy_, self.cameraMatrix_, self.distCoeffs_, mrvec, mtvec, 0.1)
                marker_rvecs.append(mrvec.ravel())
                marker_tvecs.append(mtvec.ravel())

            if len(marker_rvecs) != 4 or len(marker_tvecs) != 4:
                return

            marker_rvecs = np.array(marker_rvecs, dtype=np.float64)
            marker_tvecs = np.array(marker_tvecs, dtype=np.float64)
            rmat = Rotation.from_rotvec(marker_rvecs).mean().as_matrix()
            marker_offsets = np.array(
                [
                    [-self.delta_width_qr_center_, self.delta_height_qr_center_, 0.0],
                    [self.delta_width_qr_center_, self.delta_height_qr_center_, 0.0],
                    [self.delta_width_qr_center_, -self.delta_height_qr_center_, 0.0],
                    [-self.delta_width_qr_center_, -self.delta_height_qr_center_, 0.0],
                ],
                dtype=np.float64,
            )
            board_centers = np.array(
                [marker_tvecs[i] - (rmat @ marker_offsets[i]) for i in range(4)],
                dtype=np.float64,
            )
            tvec = board_centers.mean(axis=0).reshape(3, 1)
            logging.info("detect_qr: %s, %s, %s", tvec[0, 0], tvec[1, 0], tvec[2, 0])

            board_rvec, _ = cv2.Rodrigues(rmat.astype(np.float64))
            cv2.drawFrameAxes(self.imageCopy_, self.cameraMatrix_, self.distCoeffs_, board_rvec, tvec, 0.2)

            candidates_cloud = MAKE_POINTCLOUD()
            for center in boardCircleCenters:
                center3d = tvec[:, 0] + (rmat @ center.astype(np.float64))
                uv = self.projectPointDist(center3d.astype(np.float32), self.cameraMatrix_, self.distCoeffs_)
                cv2.circle(self.imageCopy_, (int(uv[0]), int(uv[1])), 5, (0, 255, 0), -1)
                candidates_cloud.push_back(Point(x=float(center3d[0]), y=float(center3d[1]), z=float(center3d[2])))

            groups = []
            comb(len(candidates_cloud), 4, groups)
            groups_scores = [-1.0] * len(groups)
            for i, group in enumerate(groups):
                candidates = [
                    Point(
                        x=candidates_cloud[group_index].x,
                        y=candidates_cloud[group_index].y,
                        z=candidates_cloud[group_index].z,
                    )
                    for group_index in group
                ]
                square_candidate = Square(candidates, self.delta_width_circles_, self.delta_height_circles_)
                groups_scores[i] = 1.0 if square_candidate.is_valid() else -1.0

            best_candidate_idx = -1
            best_candidate_score = -1.0
            for i, score in enumerate(groups_scores):
                if best_candidate_score == 1.0 and score == 1.0:
                    logging.error("[Mono] More than one set of candidates fit target's geometry. Please, make sure your parameters are well set. Exiting callback")
                    return
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate_idx = i

            if best_candidate_idx == -1:
                logging.warning("[Mono] Unable to find a candidate set that matches target's geometry")
                return

            for group_index in groups[best_candidate_idx]:
                centers_cloud.push_back(candidates_cloud[group_index])

            for point in centers_cloud:
                uv_circle = self.projectPointDist(np.array([point.x, point.y, point.z], dtype=np.float32), self.cameraMatrix_, self.distCoeffs_)
                cv2.circle(self.imageCopy_, (int(uv_circle[0]), int(uv_circle[1])), 2, (255, 0, 255), -1)
        else:
            logging.warning("[Mono] %d marker(s) found, 4 expected. Skipping frame...", len(ids_list))
