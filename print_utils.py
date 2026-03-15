from __future__ import annotations


def cameraIntrinsicsLines(params):
    return [
        f"Camera Model: {params.cam_model}",
        f"Image Width: {params.cam_width}",
        f"Image Height: {params.cam_height}",
        f"Focal Length X: {params.fx}",
        f"Focal Length Y: {params.fy}",
        f"Principal Point X: {params.cx}",
        f"Principal Point Y: {params.cy}",
        f"Distortion Coefficient k1: {params.k1}",
        f"Distortion Coefficient k2: {params.k2}",
        f"Distortion Coefficient p1: {params.p1}",
        f"Distortion Coefficient p2: {params.p2}",
    ]


def printCameraIntrinsics(params):
    for line in cameraIntrinsicsLines(params):
        print(line)
