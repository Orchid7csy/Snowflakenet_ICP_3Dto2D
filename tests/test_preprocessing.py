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
