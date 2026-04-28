"""Smoke tests for src.evaluation.metrics_utils (no GPU)."""

import numpy as np
from scipy.spatial.transform import Rotation

from src.evaluation.metrics_utils import (
    T_row_rigid_to_column_4x4,
    add_s,
    chamfer_distance,
    chamfer_distance_l1,
    chamfer_distance_l2,
    f_score,
    rotation_error_deg,
    se3_to_R_t,
    translation_error,
    translation_error_l2,
)


def test_chamfer_identical_zero():
    p = np.random.default_rng(0).random((50, 3))
    assert chamfer_distance_l1(p, p) < 1e-10
    assert chamfer_distance_l2(p, p) < 1e-10
    assert chamfer_distance(p, p, norm=1) < 1e-10
    assert chamfer_distance(p, p, norm=2) < 1e-10


def test_rotation_5deg_and_translation_1mm():
    rng = np.random.default_rng(1)
    R_gt = np.eye(3)
    t_gt = np.zeros(3)
    d = 5.0
    R_pred = Rotation.from_euler("z", np.deg2rad(d)).as_matrix()
    t_pred = np.array([0.0, 0.0, 0.001], dtype=np.float64)
    assert abs(rotation_error_deg(R_pred, R_gt) - d) < 0.01
    assert abs(translation_error_l2(t_pred, t_gt) - 0.001) < 1e-9


def test_T_gt_consistency():
    R = Rotation.from_euler("xyz", [0.1, -0.2, 0.05]).as_matrix()
    t = np.array([0.01, -0.02, 0.03])
    T = T_row_rigid_to_column_4x4(R, t)
    R2, t2 = se3_to_R_t(T)
    assert np.allclose(R, R2)
    assert np.allclose(t, t2)
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    col = (T[:3, :3] @ p.T).T + T[:3, 3]
    row = p @ R.T + t
    assert np.allclose(col, row)


def test_f_score_perfect_overlap():
    rng = np.random.default_rng(2)
    p = rng.random((20, 3))
    o = f_score(p, p, threshold=0.1, beta=1.0)
    assert o["precision"] == 1.0
    assert o["recall"] == 1.0
    assert abs(o["f"] - 1.0) < 1e-9


def test_add_s_identity():
    rng = np.random.default_rng(3)
    p = rng.random((30, 3))
    T = np.eye(4)
    d = add_s(p, p, T, symmetric=False)
    assert d < 1e-9


def test_translation_error_cm_scale():
    t1 = np.array([1.0, 0.0, 0.0])
    t0 = np.zeros(3)
    d = translation_error(t1, t0, unit="cm", scale_to_cm=2.5)
    assert abs(d - 2.5) < 1e-9
