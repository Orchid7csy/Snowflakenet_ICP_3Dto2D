import importlib
import os
import sys

import numpy as np
import pytest

pytest.importorskip("open3d")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = importlib.import_module("src.data.preprocessing")


def test_so3_det():
    rng = np.random.default_rng(0)
    r, t, m = p.sample_random_far_transform(rng, 0.5, 1.0)
    assert abs(np.linalg.det(r) - 1.0) < 1e-4
    assert 0.5 <= m <= 1.0


def test_inverse_pca_roundtrip():
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((100, 3)).astype(np.float32)
    p_norm, c, s = p.normalize_by_bbox(pts)
    al, r, mu = p.pca_align(p_norm, target_axis="z")
    back = p.inverse_pca(al, r, mu)
    assert back.shape == al.shape


def test_normalize_by_complete_unit_sphere():
    rng = np.random.default_rng(2)
    complete = rng.standard_normal((512, 3)).astype(np.float32) * 2.5 + np.array(
        [1.0, -2.0, 0.5], dtype=np.float32
    )
    partial = complete[:128] + 0.01
    pc, gc, c, s = p.normalize_by_complete(complete, partial)
    assert pc.shape == partial.shape
    assert gc.shape == complete.shape
    assert c.shape == (3,)
    assert s > 0
    assert float(np.linalg.norm(gc, axis=1).max()) <= 1.0 + 1e-5


def test_canonical_inverse_normalization_roundtrip():
    rng = np.random.default_rng(3)
    complete = rng.standard_normal((400, 3)).astype(np.float32)
    partial = complete[:96] + 0.005
    pc, _gc, c, s = p.normalize_by_complete(complete, partial)
    R_aug = p.random_gravity_axis_rot(rng, 25.0, axis="z")
    pc_aug = (pc @ R_aug.T).astype(np.float32)
    r_far, t_far, _ = p.sample_random_far_transform(rng, 0.5, 2.0)
    obs_w = p.apply_rigid_row(partial, r_far, t_far)
    meta = {
        "C_cano": c,
        "scale_cano": np.float32(s),
        "R_aug": R_aug,
        "R_far": r_far,
        "t_far": t_far,
    }
    back = p.apply_inverse_normalization(pc_aug, meta)
    assert back.shape == obs_w.shape
    assert float(np.max(np.abs(back - obs_w))) < 1e-3


def test_canonical_inverse_without_R_aug():
    rng = np.random.default_rng(4)
    complete = rng.standard_normal((300, 3)).astype(np.float32)
    partial = complete[:80] + 0.01
    pc, _gc, c, s = p.normalize_by_complete(complete, partial)
    r_far, t_far, _ = p.sample_random_far_transform(rng, 0.5, 1.5)
    obs_w = p.apply_rigid_row(partial, r_far, t_far)
    meta = {
        "C_cano": c,
        "scale_cano": np.float32(s),
        "R_far": r_far,
        "t_far": t_far,
    }
    back = p.apply_inverse_normalization(pc, meta)
    assert float(np.max(np.abs(back - obs_w))) < 1e-3


def test_random_gravity_axis_rot_zero_is_identity():
    rng = np.random.default_rng(5)
    R = p.random_gravity_axis_rot(rng, 0.0, axis="z")
    assert np.allclose(R, np.eye(3), atol=1e-6)
