from __future__ import annotations

import argparse

from pipeline_runner import PipelineConfig, run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cam_intrinsic_file", "--cam_intrinsic_file", default="")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--marker_size", type=float, default=-1.0)
    parser.add_argument("--delta_width_qr_center", type=float, default=-1.0)
    parser.add_argument("--delta_height_qr_center", type=float, default=-1.0)
    parser.add_argument("--delta_width_circles", type=float, default=-1.0)
    parser.add_argument("--delta_height_circles", type=float, default=-1.0)
    parser.add_argument("--min_detected_markers", type=int, default=0)
    parser.add_argument("--circle_radius", type=float, default=-1.0)
    parser.add_argument("--voxel_downsample_size", type=float, default=-1.0)
    parser.add_argument("--plane_dist_threshold", type=float, default=-1.0)
    parser.add_argument("--circle_tolerance", type=float, default=-1.0)
    parser.add_argument("--crop_min_xyz", default="")
    parser.add_argument("--crop_max_xyz", default="")
    parser.add_argument("--lidar_center_axis_map", default="")
    args = parser.parse_args()

    axis_map = None
    if args.lidar_center_axis_map:
        axis_map = tuple(item.strip() for item in args.lidar_center_axis_map.split(","))
        if len(axis_map) != 3:
            raise ValueError("--lidar_center_axis_map must contain exactly 3 comma-separated tokens")

    config = PipelineConfig(
        cam_intrinsic_file=args.cam_intrinsic_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        marker_size=args.marker_size,
        delta_width_qr_center=args.delta_width_qr_center,
        delta_height_qr_center=args.delta_height_qr_center,
        delta_width_circles=args.delta_width_circles,
        delta_height_circles=args.delta_height_circles,
        min_detected_markers=args.min_detected_markers,
        circle_radius=args.circle_radius,
        voxel_downsample_size=args.voxel_downsample_size,
        plane_dist_threshold=args.plane_dist_threshold,
        circle_tolerance=args.circle_tolerance,
        crop_min_xyz=args.crop_min_xyz,
        crop_max_xyz=args.crop_max_xyz,
        lidar_center_axis_map=axis_map,
    )
    run_pipeline(config, log_callback=print)


if __name__ == "__main__":
    main()
