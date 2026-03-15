from __future__ import annotations

import logging

import cv2
import numpy as np


def colorPointCloudWithImage(cloud, transformation, cameraMatrix, distCoeffs, image, colored_cloud):
    colored_cloud.clear()
    undistortedImage = cv2.undistort(image, cameraMatrix, distCoeffs)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    zeroDistCoeffs = np.zeros((5, 1), dtype=np.float32)

    for point in cloud:
        homogeneous_point = np.array([point.x, point.y, point.z, 1.0], dtype=np.float64)
        transformed_point = transformation @ homogeneous_point
        if transformed_point[2] < 0:
            continue

        objectPoints = np.array(
            [[[transformed_point[0], transformed_point[1], transformed_point[2]]]],
            dtype=np.float32,
        )
        imagePoints, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, zeroDistCoeffs)
        u = int(imagePoints[0, 0, 0])
        v = int(imagePoints[0, 0, 1])
        if 0 <= u < undistortedImage.shape[1] and 0 <= v < undistortedImage.shape[0]:
            color = undistortedImage[v, u]
            colored_cloud.append(
                (
                    float(transformed_point[0]),
                    float(transformed_point[1]),
                    float(transformed_point[2]),
                    int(color[2]),
                    int(color[1]),
                    int(color[0]),
                )
            )


def projectLidarOnImage(cloud, transformation, cameraMatrix, distCoeffs, image, axis_inversion=None):
    overlay = cv2.undistort(image, cameraMatrix, distCoeffs)
    projected_points = []

    for point in cloud:
        homogeneous_point = np.array([point.x, point.y, point.z, 1.0], dtype=np.float64)
        transformed_point = transformation @ homogeneous_point
        if transformed_point[2] <= 0:
            continue
        if axis_inversion is not None:
            transformed_point[:3] = axis_inversion @ transformed_point[:3]

        objectPoints = np.array(
            [[[transformed_point[0], transformed_point[1], transformed_point[2]]]],
            dtype=np.float32,
        )
        imagePoints, _ = cv2.projectPoints(
            objectPoints,
            np.zeros((3, 1), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            cameraMatrix,
            np.zeros((5, 1), dtype=np.float32),
        )
        u = int(imagePoints[0, 0, 0])
        v = int(imagePoints[0, 0, 1])
        if 0 <= u < overlay.shape[1] and 0 <= v < overlay.shape[0]:
            projected_points.append((u, v, float(transformed_point[2])))

    if not projected_points:
        return overlay

    depths = np.array([depth for _, _, depth in projected_points], dtype=np.float64)
    min_depth = float(depths.min())
    max_depth = float(depths.max())
    depth_span = max(max_depth - min_depth, 1e-6)

    for u, v, depth in projected_points:
        normalized = (depth - min_depth) / depth_span
        color = cv2.applyColorMap(
            np.array([[int((1.0 - normalized) * 255)]], dtype=np.uint8),
            cv2.COLORMAP_JET,
        )[0, 0]
        cv2.circle(overlay, (u, v), 2, tuple(int(channel) for channel in color.tolist()), -1)

    legend_height = min(240, max(120, overlay.shape[0] // 6))
    legend_width = 24
    margin = 24
    x0 = overlay.shape[1] - legend_width - margin
    y0 = margin
    legend = np.linspace(0, 255, legend_height, dtype=np.uint8).reshape(-1, 1)
    legend = np.repeat(legend, legend_width, axis=1)
    legend = cv2.applyColorMap(legend, cv2.COLORMAP_JET)
    overlay[y0 : y0 + legend_height, x0 : x0 + legend_width] = legend[::-1]
    cv2.rectangle(overlay, (x0, y0), (x0 + legend_width, y0 + legend_height), (255, 255, 255), 1)
    cv2.putText(
        overlay,
        f"{min_depth:.2f}m",
        (x0 - 70, y0 + legend_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"{max_depth:.2f}m",
        (x0 - 70, y0 + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "Depth",
        (x0 - 6, y0 + legend_height + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return overlay


def getAxisInversionMatrix(axis_name):
    matrices = {
        "x": np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        "y": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        "z": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
    }
    return matrices[axis_name]


def computeRMSE(cloud1, cloud2) -> float:
    if len(cloud1) != len(cloud2):
        logging.error("PointCloud sizes do not match, cannot compute RMSE.")
        return -1.0

    sum_sq = 0.0
    for p1, p2 in zip(cloud1, cloud2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        sum_sq += dx * dx + dy * dy + dz * dz
    return float(np.sqrt(sum_sq / len(cloud1)))
