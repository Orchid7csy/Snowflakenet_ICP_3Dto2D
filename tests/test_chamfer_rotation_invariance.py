"""
微调前必过：对同一对点云 (pred, gt) 施加同一 R ∈ SO(3) 到两边后，对称 L1 CD 与旋转前相同。

若失败，说明增强同步或 Chamfer 实现与刚体配准下距离期望不一致，须先排错再训。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.pcn_dataset import sample_rotation_matrix
from src.models.chamfer import chamfer_l1_symmetric


def _assert_r_orthonormal_so3(r: np.ndarray) -> None:
    i = r @ r.T
    assert np.allclose(i, np.eye(3), atol=1e-5, rtol=1e-5)
    assert float(np.abs(np.linalg.det(r) - 1.0)) < 1e-3


@pytest.mark.parametrize("rot_mode", ["so3", "yaw"])
def test_chamfer_l1_invariant_under_same_rotation(rot_mode: str, seed: int = 0):
    """cd(R@P, R@G) == cd(P, G)（数值容差内）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pred = torch.randn(2, 150, 3, dtype=torch.float32)
    gt = torch.randn(2, 180, 3, dtype=torch.float32)
    cd0 = chamfer_l1_symmetric(pred, gt)

    rng = np.random.default_rng(seed)
    r = sample_rotation_matrix(rot_mode, rng=rng)  # type: ignore[arg-type]
    _assert_r_orthonormal_so3(r)
    rt = torch.from_numpy(r).to(dtype=pred.dtype)
    pred_r = pred @ rt.T
    gt_r = gt @ rt.T
    cd1 = chamfer_l1_symmetric(pred_r, gt_r)

    assert torch.allclose(
        cd0, cd1, atol=1e-4, rtol=1e-3
    ), f"cd0={cd0.item():.8f} cd1={cd1.item():.8f} mode={rot_mode}"


def test_mismatching_rotation_fails_trivial_check():
    """若只对 pred 转、对 gt 不转，CD 一般不应与原来相等（不强制数值，只作反例防逻辑错误）。"""
    torch.manual_seed(1)
    pred = torch.randn(1, 100, 3, dtype=torch.float32)
    gt = torch.randn(1, 120, 3, dtype=torch.float32)
    cd0 = chamfer_l1_symmetric(pred, gt)
    r = sample_rotation_matrix("so3", rng=np.random.default_rng(2))
    rt = torch.from_numpy(r).to(dtype=pred.dtype)
    pred_r = pred @ rt.T
    cd_w = chamfer_l1_symmetric(pred_r, gt)
    # 几乎不可能偶然相等
    assert not torch.allclose(cd0, cd_w, atol=1e-6, rtol=0.0)
