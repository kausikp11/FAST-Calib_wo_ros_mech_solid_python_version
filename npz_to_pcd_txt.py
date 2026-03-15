import numpy as np
from pathlib import Path
import os

input_dir = Path("lidar_cam data/new data/tilt_left/L1_lens1/lidar/")
output_dir = Path("data_txt")
output_dir.mkdir(parents=True, exist_ok=True)

npz_files = sorted(input_dir.glob("*.npz"))

if not npz_files:
    raise RuntimeError(f"No .npz files found in {input_dir}")

def npz_to_pcd(npz_path, pcd_path, binary=False):
    print(npz_path)
    data = np.load(npz_path)

    xyz = data["xyz_m"]          # (N,3)
    intensity = data["intensity"].astype(np.float32)
    ring = data["ring"].astype(np.float32)
    distance = data["distance_m"]

    # --- Remove invalid points (important for FAST-Calib)
    valid = distance > 0.1
    xyz = xyz[valid]
    intensity = intensity[valid]
    ring = ring[valid]

    N = xyz.shape[0]

    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity ring
SIZE 4 4 4 4 4
TYPE F F F F F
COUNT 1 1 1 1 1
WIDTH {N}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {N}
DATA {"binary" if binary else "ascii"}
"""

    with open(pcd_path, "wb" if binary else "w") as f:
        if binary:
            f.write(header.encode("ascii"))
            cloud = np.column_stack((xyz, intensity, ring))
            cloud.astype(np.float32).tofile(f)
        else:
            f.write(header)
            for i in range(N):
                f.write(
                    f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} "
                    f"{intensity[i]} {ring[i]}\n"
                )

    print(f"Saved PCD with {N} points → {pcd_path}")

def npz_to_txt(npz_path, txt_path):
    print(npz_path)
    data = np.load(npz_path)

    xyz = data["xyz_m"]                 # (N,3)
    intensity = data["intensity"]
    ring = data["ring"]
    distance = data["distance_m"]

    # Remove invalid points (IMPORTANT)
    valid = distance > 0.1
    xyz = xyz[valid]
    intensity = intensity[valid]
    ring = ring[valid]

    with open(txt_path, "w") as f:
        for i in range(len(xyz)):
            f.write(
                f"{xyz[i,0]:.6f} "
                f"{xyz[i,1]:.6f} "
                f"{xyz[i,2]:.6f} "
                f"{intensity[i]} "
                f"{ring[i]}\n"
            )

    print(f"Saved {len(xyz)} points to {txt_path}")

if __name__ =='__main__':
	#os.mkdir(output_dir / f"test_fast_calib")
	for npz_path in npz_files:
		out_path = output_dir / f"test_fast_calib" /f"{npz_path.stem}.txt"
		npz_to_txt(npz_path,out_path)
