# FAST-Calib Python Verison

## Critical Note:- This is vibe-coded application based on my eariler creation in CPP - you can refer that repo from here - https://github.com/kausikp11/FAST-Calib_wo_ros_mech_solid
## Please raise a issues if you feel some error or modification required.

Contents:
- Python source for the calibration pipeline
- Desktop GUI for Linux and Windows
- Sample dataset in `data/`
- Empty `output/` folder for new runs
- Linux one-file executable in `dist/FAST-Calib-GUI-linux`

## Requirements

Tested with Python `3.10`.

Install runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Linux GUI note:

```bash
sudo apt install python3-tk
```

If you want to rebuild the executable:

```bash
pip install -r requirements-build.txt
```

## Folder Layout

```text
FAST-Calib_python_shareable/
├── data/
├── dist/
├── output/
├── gui_app.py
├── main.py
├── pipeline_runner.py
├── requirements.txt
└── ...
```

## Input Format

Each image should have a matching folder named `image_name_pcd/`.

Example:

```text
data/
├── camera_pinhole.yaml
├── image.png
└── image_pcd/
    ├── lidar_1.txt
    ├── lidar_2.npz
    └── ...
```

Supported LiDAR formats inside each `image_pcd/` folder:
- `.txt`
- `.npz`

`.txt` format:

```text
x y z intensity ring
```

`.npz` format:

```text
xyz_m
intensity
ring
distance_m
```

The loader keeps only points where:

```text
distance_m > 0.1
```

## Axis Orientation

Update the LiDAR ordering basis in `structs.py`:

```python
lidar_center_axis_map: tuple[str, str, str] = ("x", "-z", "y")
```

This means:

```text
Xcam = Xlidar
Ycam = -Zlidar
Zcam = Ylidar
```

Supported tokens:

```text
x, -x, y, -y, z, -z
```

## Run From Source

CLI:

```bash
python main.py \
  --cam_intrinsic_file data/camera_pinhole.yaml \
  --data_dir data \
  --output_dir output
```

GUI:

```bash
python gui_app.py
```

GUI features:
- change calibration parameters
- change axis orientation
- preview the projected overlay
- save and load settings
- save run details
- show or hide console logs
- create a new output folder directly from the `Create` button beside `Output Dir`

## Best Practices

Do:
- keep one image and its matching LiDAR files together using the `image_name_pcd/` naming pattern
- keep the camera YAML consistent with the actual camera used to capture the images
- start with the default parameters before tuning detection thresholds
- save GUI settings before major parameter changes so you can return to a known-good setup
- use a fresh output folder for each experiment when comparing multiple runs
- check `lidar_projection_0.png` first after every run; it is the fastest way to spot axis or transform mistakes
- change only `lidar_center_axis_map` first when the projection is mirrored, flipped, or geometrically wrong
- keep raw `.npz` files if available, even if you also export `.txt`, so you can re-run with the original data later
- use Python `3.10` for the most predictable environment match with this package

Don't:
- don't mix LiDAR files from different images inside the same `_pcd` folder
- don't rename the `_pcd` folder arbitrarily; it must match the image stem
- don't change many calibration parameters at once if you are debugging a failure
- don't treat a low RMSE alone as proof that the calibration is correct; always inspect the overlay image
- don't overwrite old outputs when you are testing different axis conventions or threshold values
- don't build the Windows `.exe` from Linux; build it on Windows with `build_windows.bat`
- don't rely on a shell with a custom ROS `PYTHONPATH` unless you intentionally want that environment involved
- don't delete the sample dataset from a shared package unless you also update the README and default GUI paths

Recommended workflow:
- first confirm the data loads and the QR image looks correct
- then verify the overlay alignment
- only after that start tuning thresholds or crop bounds

## Run Packaged Linux App

```bash
./run_gui.sh
```

or

```bash
./dist/FAST-Calib-GUI-linux
```

## Build Windows App

You need to build the Windows executable on a Windows machine. PyInstaller does not reliably cross-build a native Windows `.exe` from Linux.

On Windows:

```bat
build_windows.bat
```

That script will:
- create `.venv-win`
- install the build dependencies
- generate `dist\FAST-Calib-GUI-windows.exe`

You can then launch it with:

```bat
run_gui_windows.bat
```

Recommended Windows setup:
- Python `3.10` installed from python.org
- use `py -3.10`
- keep the `data\` folder next to the executable if you want the sample dataset available by default

## Main Outputs

Each run writes results into `output/data_pair_0/` or the output folder you choose in the GUI.

Important files:
- `lidar_projection_0.png`
- `qr_img_0.png`
- `output.txt`
- `colored_cloud_0.pcd`
- `center_cloud_0.pcd`
- `sort_centers_lidar_0.pcd`
- `sort_centers_qr_0.pcd`

## Notes

- The project does not require ROS.
- If your shell exports a custom `PYTHONPATH`, clear it before running for a cleaner standalone environment:

```bash
unset PYTHONPATH
```
