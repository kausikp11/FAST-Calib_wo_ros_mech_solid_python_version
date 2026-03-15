from __future__ import annotations

import base64
import json
import queue
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, PhotoImage, StringVar, Tk, filedialog, messagebox, simpledialog
from tkinter import scrolledtext
from tkinter import ttk

import cv2

from pipeline_runner import PipelineConfig, run_pipeline
from structs import Parameters


AXIS_TOKENS = ("x", "-x", "y", "-y", "z", "-z")


class CalibrationGui:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("FAST-Calib Desktop")
        self.root.geometry("1560x940")
        self.root.minsize(1280, 820)

        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.current_image = None
        self.run_thread = None
        self.last_result = None

        self.app_dir = self._detect_app_dir()
        self.resource_dir = self._detect_resource_dir()
        defaults = Parameters()

        default_data_dir = self._detect_default_data_dir()
        self.cam_intrinsic_file = StringVar(value=str(default_data_dir / "camera_pinhole.yaml"))
        self.data_dir = StringVar(value=str(default_data_dir))
        self.output_dir = StringVar(value=str(self.app_dir / "output_gui"))

        self.marker_size = StringVar(value=str(defaults.marker_size))
        self.delta_width_qr_center = StringVar(value=str(defaults.delta_width_qr_center))
        self.delta_height_qr_center = StringVar(value=str(defaults.delta_height_qr_center))
        self.delta_width_circles = StringVar(value=str(defaults.delta_width_circles))
        self.delta_height_circles = StringVar(value=str(defaults.delta_height_circles))
        self.min_detected_markers = StringVar(value=str(defaults.min_detected_markers))
        self.circle_radius = StringVar(value=str(defaults.circle_radius))
        self.voxel_downsample_size = StringVar(value=str(defaults.voxel_downsample_size))
        self.plane_dist_threshold = StringVar(value=str(defaults.plane_dist_threshold))
        self.circle_tolerance = StringVar(value=str(defaults.circle_tolerance))
        self.crop_min_xyz = StringVar(value=",".join(str(x) for x in defaults.crop_min_xyz))
        self.crop_max_xyz = StringVar(value=",".join(str(x) for x in defaults.crop_max_xyz))
        self.axis_x = StringVar(value=defaults.lidar_center_axis_map[0])
        self.axis_y = StringVar(value=defaults.lidar_center_axis_map[1])
        self.axis_z = StringVar(value=defaults.lidar_center_axis_map[2])

        self.status_text = StringVar(value="Idle")
        self.rmse_text = StringVar(value="--")
        self.transform_path_text = StringVar(value="--")
        self.overlay_path_text = StringVar(value="--")
        self.axis_map_text = StringVar(value=self._axis_map_value())
        self.console_button_text = StringVar(value="Show Console")
        self.console_visible = BooleanVar(value=False)

        self._configure_styles()
        self._build_layout()
        self.root.after(100, self._drain_log_queue)

    @staticmethod
    def _detect_app_dir() -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent

    @staticmethod
    def _detect_resource_dir() -> Path:
        if getattr(sys, "frozen", False):
            bundle_dir = getattr(sys, "_MEIPASS", "")
            if bundle_dir:
                return Path(bundle_dir)
        return Path(__file__).resolve().parent

    def _detect_default_data_dir(self) -> Path:
        app_data = self.app_dir / "data"
        if app_data.exists():
            return app_data
        return self.resource_dir / "data"

    def _configure_styles(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.root.configure(bg="#f3efe7")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("App.TFrame", background="#f3efe7")
        style.configure("Panel.TFrame", background="#fffaf2", relief="flat")
        style.configure("Hero.TFrame", background="#163a36")
        style.configure("HeroTitle.TLabel", background="#163a36", foreground="#fff7ea", font=("Segoe UI", 20, "bold"))
        style.configure("HeroSub.TLabel", background="#163a36", foreground="#d6e7df", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#fffaf2", foreground="#1d2a27", font=("Segoe UI", 11, "bold"))
        style.configure("Field.TLabel", background="#fffaf2", foreground="#36413d")
        style.configure("SummaryValue.TLabel", background="#fffaf2", foreground="#12201d", font=("Consolas", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Thin.TButton", font=("Segoe UI", 9))
        style.configure("TEntry", padding=6)
        style.configure("TCombobox", padding=4)
        style.configure("Card.TLabelframe", background="#fffaf2")
        style.configure("Card.TLabelframe.Label", background="#fffaf2", foreground="#1d2a27", font=("Segoe UI", 11, "bold"))

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=14, style="App.TFrame")
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        hero = ttk.Frame(main, style="Hero.TFrame", padding=18)
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        hero.columnconfigure(0, weight=1)
        ttk.Label(hero, text="FAST-Calib Desktop", style="HeroTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            hero,
            text="Tune parameters, switch axis orientation, preview projection output, and save reusable run settings.",
            style="HeroSub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        control = ttk.Frame(main, padding=(0, 0, 14, 0), style="App.TFrame")
        control.grid(row=1, column=0, sticky="ns")
        viewer = ttk.Frame(main, style="App.TFrame")
        viewer.grid(row=1, column=1, sticky="nsew")
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(1, weight=1)
        viewer.rowconfigure(2, weight=0)
        viewer.rowconfigure(3, weight=1)

        self._build_controls(control)
        self._build_viewer(viewer)

    def _build_controls(self, parent):
        parent.columnconfigure(0, weight=1)

        paths = ttk.LabelFrame(parent, text="Paths", padding=12, style="Card.TLabelframe")
        paths.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        paths.columnconfigure(1, weight=1)
        row = 0
        row = self._path_row(paths, row, "Camera YAML", self.cam_intrinsic_file, filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")], pick_dir=False)
        row = self._path_row(paths, row, "Data Dir", self.data_dir, pick_dir=True)
        self._path_row(paths, row, "Output Dir", self.output_dir, pick_dir=True, add_create_button=True)

        params = ttk.LabelFrame(parent, text="Calibration Parameters", padding=12, style="Card.TLabelframe")
        params.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        params.columnconfigure(1, weight=1)
        fields = [
            ("Marker Size", self.marker_size),
            ("Delta QR Width", self.delta_width_qr_center),
            ("Delta QR Height", self.delta_height_qr_center),
            ("Delta Circle Width", self.delta_width_circles),
            ("Delta Circle Height", self.delta_height_circles),
            ("Min Markers", self.min_detected_markers),
            ("Circle Radius", self.circle_radius),
            ("Voxel Downsample", self.voxel_downsample_size),
            ("Plane Dist Threshold", self.plane_dist_threshold),
            ("Circle Tolerance", self.circle_tolerance),
            ("Crop Min XYZ", self.crop_min_xyz),
            ("Crop Max XYZ", self.crop_max_xyz),
        ]
        for row, (label, variable) in enumerate(fields):
            ttk.Label(params, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(params, textvariable=variable, width=26).grid(row=row, column=1, sticky="ew", pady=3)

        axis = ttk.LabelFrame(parent, text="Axis Orientation", padding=12, style="Card.TLabelframe")
        axis.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        axis.columnconfigure(1, weight=1)
        self._axis_row(axis, 0, "Xcam =", self.axis_x)
        self._axis_row(axis, 1, "Ycam =", self.axis_y)
        self._axis_row(axis, 2, "Zcam =", self.axis_z)

        actions = ttk.LabelFrame(parent, text="Actions", padding=12, style="Card.TLabelframe")
        actions.grid(row=3, column=0, sticky="ew")
        for idx in range(3):
            actions.columnconfigure(idx, weight=1)
        ttk.Button(actions, text="Run Pipeline", command=self.start_run, style="Accent.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=3)
        ttk.Button(actions, text="Load Latest", command=self.load_latest_projection, style="Thin.TButton").grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(actions, text="Clear Log", command=self.clear_log, style="Thin.TButton").grid(row=0, column=2, sticky="ew", padx=(4, 0), pady=3)
        ttk.Button(actions, text="Save Settings", command=self.save_settings, style="Thin.TButton").grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=3)
        ttk.Button(actions, text="Load Settings", command=self.load_settings, style="Thin.TButton").grid(row=1, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(actions, text="Save Run Details", command=self.save_run_details, style="Thin.TButton").grid(row=1, column=2, sticky="ew", padx=(4, 0), pady=3)

    def _build_viewer(self, parent):
        summary = ttk.LabelFrame(parent, text="Run Summary", padding=12, style="Card.TLabelframe")
        summary.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        summary.columnconfigure(1, weight=1)
        self._summary_row(summary, 0, "Status", self.status_text)
        self._summary_row(summary, 1, "RMSE", self.rmse_text)
        self._summary_row(summary, 2, "Axis Map", self.axis_map_text)
        self._summary_row(summary, 3, "Overlay", self.overlay_path_text)
        self._summary_row(summary, 4, "Transform", self.transform_path_text)

        image_card = ttk.LabelFrame(parent, text="Projected Image", padding=12, style="Card.TLabelframe")
        image_card.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        image_card.columnconfigure(0, weight=1)
        image_card.rowconfigure(0, weight=1)
        self.image_label = ttk.Label(image_card, text="Run the pipeline or load an existing projection.", anchor="center", background="#f4f0e8")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        toggle_bar = ttk.Frame(parent, style="App.TFrame")
        toggle_bar.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(toggle_bar, textvariable=self.console_button_text, command=self.toggle_console, style="Thin.TButton").grid(row=0, column=0, sticky="w")

        self.console_frame = ttk.LabelFrame(parent, text="Console", padding=12, style="Card.TLabelframe")
        self.console_frame.grid(row=3, column=0, sticky="nsew")
        self.console_frame.columnconfigure(0, weight=1)
        self.console_frame.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(
            self.console_frame,
            wrap="word",
            height=15,
            bg="#101917",
            fg="#e6f3ee",
            insertbackground="#e6f3ee",
            relief="flat",
            padx=10,
            pady=10,
            font=("Consolas", 9),
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")
        self.console_frame.grid_remove()

    def _path_row(self, parent, row, label, variable, filetypes=None, pick_dir=False, add_create_button=False):
        ttk.Label(parent, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable, width=28).grid(row=row, column=1, sticky="ew", pady=3)
        ttk.Button(parent, text="Browse", style="Thin.TButton", command=lambda: self._browse(variable, filetypes=filetypes, pick_dir=pick_dir)).grid(
            row=row, column=2, sticky="ew", padx=(6, 0), pady=3
        )
        if add_create_button:
            ttk.Button(parent, text="Create", style="Thin.TButton", command=self.create_output_dir).grid(
                row=row, column=3, sticky="ew", padx=(6, 0), pady=3
            )
        return row + 1

    def _axis_row(self, parent, row, label, variable):
        ttk.Label(parent, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(parent, textvariable=variable, values=AXIS_TOKENS, width=8, state="readonly").grid(row=row, column=1, sticky="w", pady=3)

    def _summary_row(self, parent, row, label, variable):
        ttk.Label(parent, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="nw", pady=2)
        ttk.Label(parent, textvariable=variable, style="SummaryValue.TLabel", wraplength=900, justify="left").grid(row=row, column=1, sticky="w", pady=2)

    def _browse(self, variable, filetypes=None, pick_dir=False):
        current = variable.get().strip()
        base = str(Path(current).parent) if current else "."
        if pick_dir:
            selected = filedialog.askdirectory(initialdir=current or ".")
        else:
            selected = filedialog.askopenfilename(initialdir=base, filetypes=filetypes)
        if selected:
            variable.set(selected)

    def create_output_dir(self):
        current_output = Path(self.output_dir.get().strip()) if self.output_dir.get().strip() else self.app_dir / "output_gui"
        parent_dir = filedialog.askdirectory(
            title="Select parent folder for new output directory",
            initialdir=str(current_output.parent if current_output.parent.exists() else self.app_dir),
        )
        if not parent_dir:
            return

        suggested_name = current_output.name if current_output.name else "output_gui"
        folder_name = simpledialog.askstring("Create output directory", "New output folder name:", initialvalue=suggested_name, parent=self.root)
        if not folder_name:
            return

        new_output_dir = Path(parent_dir) / folder_name.strip()
        if not folder_name.strip():
            messagebox.showerror("Invalid name", "Output folder name cannot be empty.")
            return

        new_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.set(str(new_output_dir))
        self.status_text.set(f"Output directory ready: {new_output_dir}")

    def _append_log(self, message: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def toggle_console(self):
        visible = not self.console_visible.get()
        self.console_visible.set(visible)
        if visible:
            self.console_frame.grid()
            self.console_button_text.set("Hide Console")
        else:
            self.console_frame.grid_remove()
            self.console_button_text.set("Show Console")

    def start_run(self):
        if self.run_thread is not None and self.run_thread.is_alive():
            messagebox.showinfo("Run in progress", "Wait for the current run to finish.")
            return
        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.clear_log()
        self.last_result = None
        self.status_text.set("Running")
        self.rmse_text.set("--")
        self.transform_path_text.set("--")
        self.overlay_path_text.set("--")
        self.axis_map_text.set(self._axis_map_value())
        self.run_thread = threading.Thread(target=self._run_worker, args=(config,), daemon=True)
        self.run_thread.start()

    def _build_config(self):
        axis_map = (self.axis_x.get(), self.axis_y.get(), self.axis_z.get())
        output_dir = self.output_dir.get().strip()
        if not output_dir:
            raise ValueError("Output directory is required.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return PipelineConfig(
            cam_intrinsic_file=self.cam_intrinsic_file.get().strip(),
            data_dir=self.data_dir.get().strip(),
            output_dir=output_dir,
            marker_size=float(self.marker_size.get()),
            delta_width_qr_center=float(self.delta_width_qr_center.get()),
            delta_height_qr_center=float(self.delta_height_qr_center.get()),
            delta_width_circles=float(self.delta_width_circles.get()),
            delta_height_circles=float(self.delta_height_circles.get()),
            min_detected_markers=int(self.min_detected_markers.get()),
            circle_radius=float(self.circle_radius.get()),
            voxel_downsample_size=float(self.voxel_downsample_size.get()),
            plane_dist_threshold=float(self.plane_dist_threshold.get()),
            circle_tolerance=float(self.circle_tolerance.get()),
            crop_min_xyz=self.crop_min_xyz.get().strip(),
            crop_max_xyz=self.crop_max_xyz.get().strip(),
            lidar_center_axis_map=axis_map,
        )

    def _run_worker(self, config: PipelineConfig):
        try:
            result = run_pipeline(config, log_callback=lambda msg: self.log_queue.put(("log", msg)))
            self.log_queue.put(("success", result))
        except Exception:
            self.log_queue.put(("error", traceback.format_exc()))

    def _drain_log_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "success":
                    self._handle_success(payload)
                elif kind == "error":
                    self.status_text.set("Run failed")
                    self._append_log(str(payload))
                    if not self.console_visible.get():
                        self.toggle_console()
                    messagebox.showerror("Pipeline failed", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    def _handle_success(self, result):
        self.last_result = result
        results = result.get("results", [])
        self.status_text.set("Run completed")
        self.axis_map_text.set(self._axis_map_value())
        if results:
            run0 = results[0]
            self.rmse_text.set(f"{run0['rmse']:.6f}")
            self.overlay_path_text.set(str(run0["overlay_path"]))
            self.transform_path_text.set(str(run0["transform_path"]))
            self.load_image(Path(run0["overlay_path"]))
            self._write_last_run_details(run0)
        else:
            self.rmse_text.set("--")
            self.overlay_path_text.set("No output generated")
            self.transform_path_text.set("--")

    def load_latest_projection(self):
        output_dir = Path(self.output_dir.get().strip())
        projection = output_dir / "data_pair_0" / "lidar_projection_0.png"
        transform = output_dir / "data_pair_0" / "output.txt"
        if not projection.exists():
            messagebox.showinfo("No output", f"No projection found at:\n{projection}")
            return
        self.load_image(projection)
        self.overlay_path_text.set(str(projection))
        self.transform_path_text.set(str(transform) if transform.exists() else "--")
        self.axis_map_text.set(self._axis_map_value())
        self.status_text.set("Loaded latest projection")

    def load_image(self, path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")

        height, width = image.shape[:2]
        max_width, max_height = 980, 620
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1.0:
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        success, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError(f"Unable to encode image for preview: {path}")

        self.current_image = PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii"))
        self.image_label.configure(image=self.current_image, text="")

    def save_settings(self):
        path = filedialog.asksaveasfilename(
            title="Save settings",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialfile="fast_calib_settings.json",
        )
        if not path:
            return
        payload = {
            "cam_intrinsic_file": self.cam_intrinsic_file.get().strip(),
            "data_dir": self.data_dir.get().strip(),
            "output_dir": self.output_dir.get().strip(),
            "marker_size": self.marker_size.get().strip(),
            "delta_width_qr_center": self.delta_width_qr_center.get().strip(),
            "delta_height_qr_center": self.delta_height_qr_center.get().strip(),
            "delta_width_circles": self.delta_width_circles.get().strip(),
            "delta_height_circles": self.delta_height_circles.get().strip(),
            "min_detected_markers": self.min_detected_markers.get().strip(),
            "circle_radius": self.circle_radius.get().strip(),
            "voxel_downsample_size": self.voxel_downsample_size.get().strip(),
            "plane_dist_threshold": self.plane_dist_threshold.get().strip(),
            "circle_tolerance": self.circle_tolerance.get().strip(),
            "crop_min_xyz": self.crop_min_xyz.get().strip(),
            "crop_max_xyz": self.crop_max_xyz.get().strip(),
            "lidar_center_axis_map": [self.axis_x.get(), self.axis_y.get(), self.axis_z.get()],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status_text.set(f"Settings saved: {path}")

    def load_settings(self):
        path = filedialog.askopenfilename(
            title="Load settings",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.cam_intrinsic_file.set(payload.get("cam_intrinsic_file", self.cam_intrinsic_file.get()))
        self.data_dir.set(payload.get("data_dir", self.data_dir.get()))
        self.output_dir.set(payload.get("output_dir", self.output_dir.get()))
        self.marker_size.set(payload.get("marker_size", self.marker_size.get()))
        self.delta_width_qr_center.set(payload.get("delta_width_qr_center", self.delta_width_qr_center.get()))
        self.delta_height_qr_center.set(payload.get("delta_height_qr_center", self.delta_height_qr_center.get()))
        self.delta_width_circles.set(payload.get("delta_width_circles", self.delta_width_circles.get()))
        self.delta_height_circles.set(payload.get("delta_height_circles", self.delta_height_circles.get()))
        self.min_detected_markers.set(payload.get("min_detected_markers", self.min_detected_markers.get()))
        self.circle_radius.set(payload.get("circle_radius", self.circle_radius.get()))
        self.voxel_downsample_size.set(payload.get("voxel_downsample_size", self.voxel_downsample_size.get()))
        self.plane_dist_threshold.set(payload.get("plane_dist_threshold", self.plane_dist_threshold.get()))
        self.circle_tolerance.set(payload.get("circle_tolerance", self.circle_tolerance.get()))
        self.crop_min_xyz.set(payload.get("crop_min_xyz", self.crop_min_xyz.get()))
        self.crop_max_xyz.set(payload.get("crop_max_xyz", self.crop_max_xyz.get()))
        axis_map = payload.get("lidar_center_axis_map", [self.axis_x.get(), self.axis_y.get(), self.axis_z.get()])
        if len(axis_map) == 3:
            self.axis_x.set(axis_map[0])
            self.axis_y.set(axis_map[1])
            self.axis_z.set(axis_map[2])
        self.axis_map_text.set(self._axis_map_value())
        self.status_text.set(f"Settings loaded: {path}")

    def save_run_details(self):
        if self.last_result is None or not self.last_result.get("results"):
            messagebox.showinfo("No run details", "Run the pipeline first, then save the details.")
            return
        path = filedialog.asksaveasfilename(
            title="Save run details",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialfile="fast_calib_run_details.json",
        )
        if not path:
            return
        run0 = self.last_result["results"][0]
        payload = self._run_details_payload(run0)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status_text.set(f"Run details saved: {path}")

    def _write_last_run_details(self, run0):
        output_dir = Path(self.output_dir.get().strip())
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self._run_details_payload(run0)
        Path(output_dir / "last_run_details.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _run_details_payload(self, run0):
        return {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "status": self.status_text.get(),
            "rmse": run0["rmse"],
            "overlay_path": str(run0["overlay_path"]),
            "transform_path": str(run0["transform_path"]),
            "cam_intrinsic_file": self.cam_intrinsic_file.get().strip(),
            "data_dir": self.data_dir.get().strip(),
            "output_dir": self.output_dir.get().strip(),
            "marker_size": self.marker_size.get().strip(),
            "delta_width_qr_center": self.delta_width_qr_center.get().strip(),
            "delta_height_qr_center": self.delta_height_qr_center.get().strip(),
            "delta_width_circles": self.delta_width_circles.get().strip(),
            "delta_height_circles": self.delta_height_circles.get().strip(),
            "min_detected_markers": self.min_detected_markers.get().strip(),
            "circle_radius": self.circle_radius.get().strip(),
            "voxel_downsample_size": self.voxel_downsample_size.get().strip(),
            "plane_dist_threshold": self.plane_dist_threshold.get().strip(),
            "circle_tolerance": self.circle_tolerance.get().strip(),
            "crop_min_xyz": self.crop_min_xyz.get().strip(),
            "crop_max_xyz": self.crop_max_xyz.get().strip(),
            "lidar_center_axis_map": [self.axis_x.get(), self.axis_y.get(), self.axis_z.get()],
        }

    def _axis_map_value(self):
        return f"Xcam={self.axis_x.get()}, Ycam={self.axis_y.get()}, Zcam={self.axis_z.get()}"


def main():
    root = Tk()
    CalibrationGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
