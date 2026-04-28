#!/usr/bin/env python3
"""
交互式可视化：调节 transforms.apply_depth_dropout（强背光）与 apply_specular_dropout（强反光），
并在 Open3D Filament 场景中绘制相机、远端背光球/方向光、或点光源与反射示意线。

用法:
  python scripts/06_viz_lighting_dropout.py -i data/processed/.../test/gt/xxx.npy
  python scripts/06_viz_lighting_dropout.py -i sample.pcd --no-hpr
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.transforms import (  # noqa: E402
    apply_depth_dropout,
    apply_specular_dropout,
    simulate_rgbd_single_view,
)


def _depth_keep_indices(
    points: np.ndarray,
    camera_pos: np.ndarray,
    missing_rate: float,
    noise_scale: float,
) -> np.ndarray:
    """与 src/data/transforms.apply_depth_dropout 的索引选择一致（须同 RNG 状态调用）。"""
    missing_rate = float(np.clip(missing_rate, 0.0, 0.95))
    if missing_rate <= 0.0:
        return np.arange(len(points), dtype=np.int64)
    num_points = len(points)
    num_keep = int(num_points * (1.0 - missing_rate))
    distances = np.linalg.norm(points - camera_pos, axis=1)
    mx = distances.max()
    if mx <= 0:
        mx = 1.0
    distances_normalized = distances / mx
    noise = np.random.normal(loc=0.0, scale=float(noise_scale), size=num_points)
    perturbed = distances_normalized + noise
    sorted_indices = np.argsort(perturbed)
    return sorted_indices[:num_keep].astype(np.int64)


def _specular_keep_indices(
    points: np.ndarray,
    normals: np.ndarray,
    camera_pos: np.ndarray,
    light_dir: np.ndarray,
    missing_rate: float,
    noise_scale: float,
    specular_exponent: float,
) -> np.ndarray:
    """与 apply_specular_dropout 的索引选择一致（须同 RNG 状态调用）。"""
    if missing_rate <= 0.0 or len(points) == 0:
        return np.arange(len(points), dtype=np.int64)
    num_points = len(points)
    num_drop = int(num_points * missing_rate)
    num_keep = num_points - num_drop
    V = camera_pos - points
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    L = np.asarray(light_dir, dtype=np.float32)
    L = L / np.linalg.norm(L)
    H = L + V
    H = H / np.linalg.norm(H, axis=1, keepdims=True)
    specular_intensity = np.sum(normals * H, axis=1)
    specular_intensity = np.clip(specular_intensity, 0, 1)
    k = max(float(specular_exponent), 1e-6)
    specular_rank = specular_intensity**k
    noise = np.random.normal(loc=0.0, scale=float(noise_scale), size=num_points)
    perturbed = specular_rank + noise
    sorted_indices = np.argsort(perturbed)
    return sorted_indices[:num_keep].astype(np.int64)


def _verify_index_sync_optional() -> None:
    if os.environ.get("LIGHTING_DROPOUT_VERIFY", "") != "1":
        return
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((200, 3)).astype(np.float32)
    nml = rng.standard_normal((200, 3)).astype(np.float32)
    nml /= np.linalg.norm(nml, axis=1, keepdims=True) + 1e-8
    cam = np.array([1.0, 0.5, 2.0], np.float32)
    for seed in (1, 42, 99):
        np.random.seed(seed)
        k1, _ = apply_depth_dropout(pts, nml, cam, 0.3, noise_scale=0.1)
        np.random.seed(seed)
        idx = _depth_keep_indices(pts, cam, 0.3, 0.1)
        k2 = pts[idx].astype(np.float32)
        if not np.allclose(k1, k2, atol=1e-4):
            raise RuntimeError("depth dropout 索引与 transforms 不一致，请同步脚本。")


def load_points(path: Path) -> np.ndarray:
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(str(path))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"期望 (N,3) .npy，得到 {arr.shape}")
        return arr.astype(np.float32)
    import open3d as o3d

    pc = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pc.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError(f"空点云: {path}")
    return pts.astype(np.float32)


def normalize_to_unit_sphere(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """居中并缩放到单位球；返回归一化点与半径（用于逆变换，本脚本未用）。"""
    c = pts.mean(axis=0)
    x = pts - c
    r = float(np.max(np.linalg.norm(x, axis=1)) or 1.0)
    y = (x / r).astype(np.float32)
    return y, r


def sph_to_point(center: np.ndarray, r: float, theta: float, phi: float) -> np.ndarray:
    """theta: 方位角 xy, phi: 仰角。"""
    cp = np.cos(phi)
    return center + r * np.array(
        [cp * np.cos(theta), cp * np.sin(theta), np.sin(phi)], dtype=np.float32
    )


def frustum_lineset(cam: np.ndarray, target: np.ndarray, size: float = 0.08):
    import open3d as o3d

    fwd = target - cam
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(fwd, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(fwd, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, fwd)
    up = up / (np.linalg.norm(up) + 1e-8)
    near = cam + fwd * 0.15
    hw = size
    p0 = near + (-right - up) * hw
    p1 = near + (right - up) * hw
    p2 = near + (right + up) * hw
    p3 = near + (-right + up) * hw
    pts = np.vstack([cam, p0, p1, p2, p3])
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32)),
    )
    ls.paint_uniform_color([0.2, 0.6, 0.95])
    return ls


def run_app(
    points: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    *,
    use_hpr: bool,
    init_seed: int,
) -> None:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    _verify_index_sync_optional()

    def _make_lit_material(
        base_color: list[float],
        *,
        metallic: float | None = None,
        roughness: float | None = None,
    ):
        """defaultLit 材质；旧版 pybind（如部分 CUDA 构建）无 metallic/roughness 字段时跳过。"""
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.base_color = list(base_color)
        for attr, val in (("metallic", metallic), ("roughness", roughness)):
            if val is None:
                continue
            if hasattr(mat, attr):
                setattr(mat, attr, float(val))
        return mat

    class LightingDropoutWindow:
        MODE_BACK = 0
        MODE_SPEC = 1

        def __init__(self) -> None:
            self._points = np.asarray(points, dtype=np.float32)
            self._normals = np.asarray(normals, dtype=np.float32)
            self._center = np.asarray(center, dtype=np.float32)
            self._use_hpr = use_hpr
            self._seed = init_seed

            self._mode = self.MODE_BACK
            self._missing = 0.3
            self._noise = 0.1
            self._cam_r = 2.2
            self._cam_theta = 0.7
            self._cam_phi = 0.35
            self._spec_exp = 2.0
            self._lt_theta = -0.8
            self._lt_phi = 0.25
            self._lt_dist = 2.5
            self._show_dropped = True
            self._show_rays = True
            self._show_normals = False
            self._back_strength = 0.5
            self._active_light_name: str | None = None

            gui.Application.instance.initialize()
            self._win = gui.Application.instance.create_window("Lighting dropout viz", 1280, 800)
            self._win.set_on_layout(self._on_layout)
            self._win.set_on_close(self._on_close)

            self._scene = gui.SceneWidget()
            self._scene.scene = rendering.Open3DScene(self._win.renderer)
            self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
            self._scene.scene.set_background([0.12, 0.12, 0.14, 1.0])
            self._scene.scene.scene.enable_sun_light(False)

            em = 8
            self._panel = gui.Vert(0, gui.Margins(em, em, em, em))

            self._status = gui.Label("…")
            self._panel.add_child(self._status)

            self._mode_combo = gui.Combobox()
            self._mode_combo.add_item("Backlight (depth dropout)")
            self._mode_combo.add_item("Specular (specular dropout)")
            self._mode_combo.set_on_selection_changed(self._on_mode)
            self._panel.add_child(gui.Label("Mode"))
            self._panel.add_child(self._mode_combo)

            self._missing_slider, self._missing_label = self._make_slider(
                self._panel, "missing_rate", 0.0, 0.95, self._missing, self._on_missing
            )
            self._noise_slider, self._noise_label = self._make_slider(
                self._panel, "noise_scale", 0.0, 0.3, self._noise, self._on_noise
            )
            self._cam_r_slider, _ = self._make_slider(
                self._panel, "cam_distance", 0.8, 5.0, self._cam_r, self._on_cam
            )
            self._cam_th_slider, _ = self._make_slider(
                self._panel, "cam_theta", -3.15, 3.15, self._cam_theta, self._on_cam
            )
            self._cam_ph_slider, _ = self._make_slider(
                self._panel, "cam_phi", -1.4, 1.4, self._cam_phi, self._on_cam
            )

            self._back_grid = gui.Vert(0, gui.Margins(0, em, 0, 0))
            self._back_grid.add_child(gui.Label("-- Backlight --"))
            self._back_grid.add_child(
                gui.Label("back_strength: stronger sun -> wider depth-cutoff noise")
            )
            self._sun_far_slider, _ = self._make_slider(
                self._back_grid, "back_strength", 0.0, 1.0, self._back_strength, self._on_back_param
            )
            self._panel.add_child(self._back_grid)

            self._spec_grid = gui.Vert(0, gui.Margins(0, em, 0, 0))
            self._spec_grid.add_child(gui.Label("-- Specular --"))
            self._sp_exp_slider, _ = self._make_slider(
                self._spec_grid, "specular_exponent", 0.5, 8.0, self._spec_exp, self._on_spec_param
            )
            self._lt_th_slider, _ = self._make_slider(
                self._spec_grid, "light_theta", -3.15, 3.15, self._lt_theta, self._on_spec_param
            )
            self._lt_ph_slider, _ = self._make_slider(
                self._spec_grid, "light_phi", -1.4, 1.4, self._lt_phi, self._on_spec_param
            )
            self._lt_dist_slider, _ = self._make_slider(
                self._spec_grid, "light_distance", 0.5, 6.0, self._lt_dist, self._on_spec_param
            )
            self._panel.add_child(self._spec_grid)

            self._chk_drop = gui.Checkbox("show dropped points")
            self._chk_drop.checked = True
            self._chk_drop.set_on_checked(self._on_chk_drop)
            self._panel.add_child(self._chk_drop)

            self._chk_rays = gui.Checkbox("show light rays")
            self._chk_rays.checked = True
            self._chk_rays.set_on_checked(self._on_chk_rays)
            self._panel.add_child(self._chk_rays)

            self._chk_norm = gui.Checkbox("show normals (debug)")
            self._chk_norm.checked = False
            self._chk_norm.set_on_checked(self._on_chk_norm)
            self._panel.add_child(self._chk_norm)

            rng = gui.Horiz(0, gui.Margins(0, em, 0, 0))
            b_rand = gui.Button("randomize seed")
            b_rand.set_on_clicked(self._on_rand_seed)
            b_reset = gui.Button("reset")
            b_reset.set_on_clicked(self._on_reset)
            rng.add_child(b_rand)
            rng.add_child(b_reset)
            self._panel.add_child(rng)

            b_save = gui.Button("save preset JSON")
            b_save.set_on_clicked(self._on_save_preset)
            self._panel.add_child(b_save)

            self._win.add_child(self._scene)
            self._win.add_child(self._panel)

            self._on_mode(self._mode_combo.selected_text, int(self._mode_combo.selected_index))
            self._setup_camera()

        def _make_slider(
            self,
            parent: gui.Widget,
            title: str,
            vmin: float,
            vmax: float,
            val: float,
            cb,
        ) -> tuple[gui.Slider, gui.Label]:
            grid = gui.Vert(0, gui.Margins(0, 4, 0, 0))
            lab = gui.Label(f"{title}: {val:.4f}")
            sl = gui.Slider(gui.Slider.DOUBLE)
            sl.set_limits(vmin, vmax)
            sl.double_value = val
            sl.set_on_value_changed(lambda v: cb(title, lab, sl, v))
            grid.add_child(lab)
            grid.add_child(sl)
            parent.add_child(grid)
            return sl, lab

        def _on_layout(self, ctx) -> None:
            r = self._win.content_rect
            self._panel.frame = gui.Rect(r.x, r.y, 320, r.height)
            self._scene.frame = gui.Rect(r.x + 320, r.y, r.width - 320, r.height)

        def _on_close(self) -> bool:
            gui.Application.instance.quit()
            return True

        def _camera_pos(self) -> np.ndarray:
            return sph_to_point(self._center, self._cam_r, self._cam_theta, self._cam_phi)

        def _light_pos(self) -> np.ndarray:
            return sph_to_point(self._center, self._lt_dist, self._lt_theta, self._lt_phi)

        def _light_dir_for_transform(self) -> np.ndarray:
            lp = self._light_pos()
            d = lp - self._center
            n = np.linalg.norm(d)
            if n < 1e-6:
                return np.array([1.0, -1.0, 0.5], dtype=np.float32)
            return (d / n).astype(np.float32)

        def _on_mode(self, *args) -> None:
            # Open3D: (text, index) 或 (combobox, text, index)
            index = int(args[-1])
            self._mode = index
            self._back_grid.visible = self._mode == self.MODE_BACK
            self._spec_grid.visible = self._mode == self.MODE_SPEC
            try:
                self._win.set_needs_layout()
            except Exception:
                pass
            self._refresh_scene()

        def _on_missing(self, title, lab, sl, v) -> None:
            self._missing = float(v)
            lab.text = f"{title}: {v:.4f}"
            self._refresh_scene()

        def _on_noise(self, title, lab, sl, v) -> None:
            self._noise = float(v)
            lab.text = f"{title}: {v:.4f}"
            self._refresh_scene()

        def _on_cam(self, title, lab, sl, v) -> None:
            if "distance" in title:
                self._cam_r = float(v)
            elif "theta" in title:
                self._cam_theta = float(v)
            else:
                self._cam_phi = float(v)
            lab.text = f"{title}: {float(v):.4f}"
            self._refresh_scene()

        def _on_spec_param(self, title, lab, sl, v) -> None:
            if "exponent" in title:
                self._spec_exp = float(v)
            elif "light_theta" in title:
                self._lt_theta = float(v)
            elif "light_phi" in title:
                self._lt_phi = float(v)
            elif "light_distance" in title:
                self._lt_dist = float(v)
            lab.text = f"{title}: {float(v):.4f}"
            self._refresh_scene()

        def _on_back_param(self, title, lab, sl, v) -> None:
            self._back_strength = float(v)
            lab.text = f"{title}: {float(v):.4f}"
            self._refresh_scene()

        def _on_chk_drop(self, checked: bool) -> None:
            self._show_dropped = checked
            self._refresh_scene()

        def _on_chk_rays(self, checked: bool) -> None:
            self._show_rays = checked
            self._refresh_scene()

        def _on_chk_norm(self, checked: bool) -> None:
            self._show_normals = checked
            self._refresh_scene()

        def _on_rand_seed(self) -> None:
            self._seed = int(np.random.randint(0, 2**31 - 1))
            self._refresh_scene()

        def _on_reset(self) -> None:
            self._missing = 0.3
            self._noise = 0.1
            self._refresh_scene()

        def _on_save_preset(self) -> None:
            p = Path(_PROJECT_ROOT) / "data" / "processed" / "lighting_dropout_preset.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            d = {
                "seed": self._seed,
                "mode": "backlight" if self._mode == self.MODE_BACK else "specular",
                "missing_rate": self._missing,
                "noise_scale": self._noise,
                "cam": {
                    "r": self._cam_r,
                    "theta": self._cam_theta,
                    "phi": self._cam_phi,
                },
                "backlight": {"sun_far": self._sun_far},
                "specular": {
                    "exponent": self._spec_exp,
                    "light_theta": self._lt_theta,
                    "light_phi": self._lt_phi,
                    "light_distance": self._lt_dist,
                },
            }
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")
            self._status.text = f"saved preset to {p}"

        def _base_cloud(self) -> tuple[np.ndarray, np.ndarray]:
            pts = self._points
            nml = self._normals
            if self._use_hpr:
                cam = self._camera_pos()
                pts, nml = simulate_rgbd_single_view(pts, camera_pos=cam)
            return pts, nml

        def _refresh_scene(self) -> None:
            import open3d as o3d

            scene = self._scene.scene
            for name in (
                "kept",
                "dropped",
                "cam_frustum",
                "sun_sphere",
                "sun_rays",
                "light_bulb",
                "spec_rays",
                "normals_ls",
            ):
                try:
                    scene.remove_geometry(name)
                except Exception:
                    pass

            if self._active_light_name:
                try:
                    scene.scene.remove_light(self._active_light_name)
                except Exception:
                    pass
                self._active_light_name = None

            cam = self._camera_pos()
            pts, nml = self._base_cloud()
            n0 = len(pts)
            if n0 == 0:
                self._status.text = "empty cloud (try --no-hpr)"
                return
            np.random.seed(self._seed)

            # back_strength: physical coupling for backlight mode
            #   stronger sun -> larger noise band on the depth-cutoff -> dropouts
            #   leak into mid-range, not just far points. missing_rate stays the
            #   master quota slider.
            eff_noise = float(self._noise)
            if self._mode == self.MODE_BACK:
                eff_noise = float(np.clip(self._noise + 0.25 * self._back_strength, 0.0, 1.0))
                kidx = _depth_keep_indices(pts, cam, self._missing, eff_noise)
                np.random.seed(self._seed)
                kpts, kn = apply_depth_dropout(
                    pts, nml, cam, self._missing, noise_scale=eff_noise
                )
            else:
                ldir = self._light_dir_for_transform()
                kidx = _specular_keep_indices(
                    pts,
                    nml,
                    cam,
                    ldir,
                    self._missing,
                    self._noise,
                    self._spec_exp,
                )
                np.random.seed(self._seed)
                kpts, kn = apply_specular_dropout(
                    pts,
                    nml,
                    camera_pos=cam,
                    light_dir=ldir,
                    missing_rate=self._missing,
                    noise_scale=self._noise,
                    specular_exponent=self._spec_exp,
                )

            if kpts.shape[0] != len(kidx):
                raise RuntimeError(
                    f"internal: kept {kpts.shape[0]} != idx {len(kidx)}"
                )
            all_idx = np.arange(n0)
            drop_idx = np.setdiff1d(all_idx, kidx, assume_unique=False)
            dpts = pts[drop_idx] if len(drop_idx) else np.zeros((0, 3), np.float32)

            nk = len(kidx)
            miss_actual = 1.0 - (nk / max(n0, 1))
            extra = ""
            if self._mode == self.MODE_BACK:
                extra = f" | back_strength={self._back_strength:.2f} eff_noise={eff_noise:.3f}"
            self._status.text = (
                f"seed={self._seed} | kept={nk}/{n0} ({100 * nk / max(n0, 1):.1f}%) "
                f"| miss={miss_actual:.3f} target~{self._missing:.3f}{extra}"
            )

            # Kept point cloud colors
            if self._mode == self.MODE_BACK:
                dist = np.linalg.norm(kpts - cam, axis=1)
                dmx = dist.max() or 1.0
                t = (dist / dmx).reshape(-1, 1)
                cols = (1.0 - t) * np.array([[0.2, 0.45, 0.95]]) + t * np.array(
                    [[0.95, 0.85, 0.35]]
                )
            else:
                cols = np.tile(np.array([[0.85, 0.2, 0.65]]), (len(kpts), 1))

            p_kept = o3d.geometry.PointCloud()
            p_kept.points = o3d.utility.Vector3dVector(kpts.astype(np.float64))
            p_kept.normals = o3d.utility.Vector3dVector(kn.astype(np.float64))
            p_kept.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

            if self._mode == self.MODE_SPEC:
                mat_k = _make_lit_material(
                    [0.85, 0.85, 0.88, 1.0], metallic=0.55, roughness=0.28
                )
            else:
                mat_k = _make_lit_material(
                    [0.75, 0.75, 0.8, 1.0], metallic=0.15, roughness=0.55
                )

            scene.add_geometry("kept", p_kept, mat_k)

            if self._show_dropped and len(dpts) > 0:
                p_drop = o3d.geometry.PointCloud()
                p_drop.points = o3d.utility.Vector3dVector(dpts.astype(np.float64))
                # Distinct from kept clouds: red-ish for backlight (lost in glare),
                # grey for specular (lost in highlight). Both unlit and translucent.
                drop_rgb = [0.95, 0.25, 0.25] if self._mode == self.MODE_BACK else [0.55, 0.55, 0.6]
                p_drop.paint_uniform_color(drop_rgb)
                mat_d = rendering.MaterialRecord()
                mat_d.shader = "defaultUnlit"
                mat_d.base_color = drop_rgb + [0.45]
                if hasattr(mat_d, "point_size"):
                    mat_d.point_size = 1.5
                scene.add_geometry("dropped", p_drop, mat_d)

            mat_fr = rendering.MaterialRecord()
            mat_fr.shader = "defaultUnlit"
            mat_fr.base_color = [0.2, 0.65, 1.0, 1.0]
            scene.add_geometry("cam_frustum", frustum_lineset(cam, self._center), mat_fr)

            if self._mode == self.MODE_BACK:
                view_vec = self._center - cam
                view_vec = view_vec / (np.linalg.norm(view_vec) + 1e-8)
                # Sun sits at a fixed visual distance behind the object; its
                # *radius* and color encode back_strength so the user gets a
                # direct cue that the sun got "stronger".
                sun_pos = self._center + view_vec * 3.5
                sun_r = 0.08 + 0.30 * float(self._back_strength)
                sun = o3d.geometry.TriangleMesh.create_sphere(radius=float(sun_r))
                sun.translate(sun_pos.astype(np.float64))
                col_sun = [
                    1.0,
                    0.45 - 0.25 * float(self._back_strength),
                    0.15 - 0.10 * float(self._back_strength),
                ]
                sun.paint_uniform_color([float(c) for c in col_sun])
                mat_sun = rendering.MaterialRecord()
                mat_sun.shader = "defaultUnlit"
                mat_sun.base_color = [float(col_sun[0]), float(col_sun[1]), float(col_sun[2]), 1.0]
                scene.add_geometry("sun_sphere", sun, mat_sun)

                # Directional light: from sun toward object
                sun_dir = (self._center - sun_pos).astype(np.float32)
                sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-8)
                col = np.array([[1.0], [0.85], [0.65]], dtype=np.float32)
                intensity = float(40000.0 + 90000.0 * self._back_strength)
                scene.scene.add_directional_light(
                    "dir_back", col, sun_dir.reshape(3, 1), intensity, True
                )
                self._active_light_name = "dir_back"

                if self._show_rays and len(pts) > 0:
                    ray_rng = np.random.default_rng(int(self._seed) + 404)
                    ray_pts = []
                    ray_lines = []
                    vi = 0
                    for _ in range(5):
                        j = int(ray_rng.integers(0, len(pts)))
                        a = sun_pos
                        b = pts[j]
                        ray_pts.extend([a, b])
                        ray_lines.append([vi, vi + 1])
                        vi += 2
                    if ray_pts:
                        ls = o3d.geometry.LineSet(
                            points=o3d.utility.Vector3dVector(np.asarray(ray_pts, dtype=np.float64)),
                            lines=o3d.utility.Vector2iVector(np.asarray(ray_lines, dtype=np.int32)),
                        )
                        ls.paint_uniform_color([1.0, 0.5, 0.2])
                        mat_r = rendering.MaterialRecord()
                        mat_r.shader = "defaultUnlit"
                        mat_r.base_color = [1.0, 0.55, 0.25, 1.0]
                        scene.add_geometry("sun_rays", ls, mat_r)
            else:
                lp = self._light_pos()
                bulb = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
                bulb.translate(lp.astype(np.float64))
                bulb.paint_uniform_color([1.0, 0.95, 0.3])
                mat_b = rendering.MaterialRecord()
                mat_b.shader = "defaultUnlit"
                mat_b.base_color = [1.0, 0.95, 0.4, 1.0]
                scene.add_geometry("light_bulb", bulb, mat_b)

                colp = np.array([[1.0], [1.0], [0.85]], dtype=np.float32)
                pos = lp.astype(np.float32).reshape(3, 1)
                scene.scene.add_point_light("point_spec", colp, pos, 2.5e6, 120.0, True)
                self._active_light_name = "point_spec"

                if self._show_rays and len(pts) > 0:
                    ray_rng = np.random.default_rng(int(self._seed) + 808)
                    ray_pts = []
                    ray_lines = []
                    vi = 0
                    L = self._light_dir_for_transform().astype(np.float64)
                    for _ in range(6):
                        j = int(ray_rng.integers(0, len(pts)))
                        p = pts[j]
                        n = nml[j].astype(np.float64)
                        n = n / (np.linalg.norm(n) + 1e-8)
                        v_in = L
                        ref = v_in - 2 * np.dot(v_in, n) * n
                        ref = ref / (np.linalg.norm(ref) + 1e-8)
                        mid = p + n * 0.04
                        end = p + ref * 0.25
                        ray_pts.extend([lp.astype(np.float64), p, mid, end])
                        ray_lines.extend([[vi, vi + 1], [vi + 1, vi + 2], [vi + 2, vi + 3]])
                        vi += 4
                    ls = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(np.asarray(ray_pts, dtype=np.float64)),
                        lines=o3d.utility.Vector2iVector(np.asarray(ray_lines, dtype=np.int32)),
                    )
                    ls.paint_uniform_color([1.0, 0.88, 0.25])
                    mat_r = rendering.MaterialRecord()
                    mat_r.shader = "defaultUnlit"
                    mat_r.base_color = [1.0, 0.9, 0.3, 1.0]
                    scene.add_geometry("spec_rays", ls, mat_r)

            if self._show_normals and len(kpts) > 0:
                step = max(1, len(kpts) // 400)
                subs = kpts[::step]
                sn = kn[::step]
                ends = subs + sn * 0.06
                rp = []
                ln_idx = []
                vi = 0
                for a, b in zip(subs, ends):
                    rp.extend([a, b])
                    ln_idx.append([vi, vi + 1])
                    vi += 2
                ls = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(np.asarray(rp, dtype=np.float64)),
                    lines=o3d.utility.Vector2iVector(np.asarray(ln_idx, dtype=np.int32)),
                )
                ls.paint_uniform_color([0.3, 1.0, 0.4])
                mat_n = rendering.MaterialRecord()
                mat_n.shader = "defaultUnlit"
                scene.add_geometry("normals_ls", ls, mat_n)

            scene.show_axes(True)

        def _setup_camera(self) -> None:
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60.0, bounds, bounds.get_center())

        def run(self) -> None:
            gui.Application.instance.run()

    LightingDropoutWindow().run()


def main() -> int:
    ap = argparse.ArgumentParser(description="交互式光照缺失可视化 (Open3D GUI)")
    ap.add_argument("-i", "--input", type=str, required=True, help=".npy (N,3) 或 .pcd/.ply")
    ap.add_argument("--no-hpr", action="store_true", help="不做单视 HPR，直接对全体点调 dropout")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    path = Path(args.input).expanduser()
    if not path.is_file():
        print(f"文件不存在: {path}", file=sys.stderr)
        return 1

    pts = load_points(path)
    pts, _r = normalize_to_unit_sphere(pts)
    center = pts.mean(axis=0).astype(np.float32)

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=40)
    )
    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        pass
    normals = np.asarray(pcd.normals, dtype=np.float32)

    run_app(pts, normals, center, use_hpr=not args.no_hpr, init_seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
