"""Snowflake 补全数据集与 PCN 旋转增强。"""
from __future__ import annotations

import os

import numpy as np
import torch


def sample_rotation_matrix(mode: str, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if mode == "signflip":
        # PCA 主轴符号二义性：独立 ±1，保持 det(R)=+1（右手系）
        s = rng.choice([-1.0, 1.0], size=3).astype(np.float32)
        if float(np.prod(s)) < 0:
            s[0] = -s[0]
        return np.diag(s).astype(np.float32)
    if mode == "yaw":
        theta = float(rng.uniform(0, 2 * np.pi))
        c, s = np.cos(theta), np.sin(theta)
        return np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32,
        )
    a = rng.standard_normal((3, 3)).astype(np.float64)
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q.astype(np.float32)


class SnowflakeDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="train", num_points=2048, transform=None):
        self.input_path = os.path.join(root_dir, split, "input")
        self.gt_path = os.path.join(root_dir, split, "gt")
        self.file_list = [f for f in os.listdir(self.input_path) if f.endswith(".npy")]
        self.num_points = num_points
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.file_list)

    def _resample(self, pcd, n):
        curr_n = pcd.shape[0]
        if curr_n == n:
            return pcd
        if curr_n > n:
            idx = np.random.choice(curr_n, n, replace=False)
        else:
            idx = np.concatenate([
                np.arange(curr_n),
                np.random.choice(curr_n, n - curr_n, replace=True),
            ])
        return pcd[idx]

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        input_pcd = np.load(os.path.join(self.input_path, file_name))
        gt_pcd = np.load(os.path.join(self.gt_path, file_name))
        input_pcd = self._resample(input_pcd, self.num_points)
        gt_pcd = self._resample(gt_pcd, self.num_points)
        input_tensor = torch.from_numpy(input_pcd).float()
        gt_tensor = torch.from_numpy(gt_pcd).float()
        if self.transform:
            input_tensor, gt_tensor = self.transform(input_tensor, gt_tensor)
        return input_tensor, gt_tensor


class CompletionDataset(SnowflakeDataset):
    def __init__(self, root_dir, split="train", input_points=2048, gt_points=16384):
        super().__init__(root_dir, split=split, num_points=input_points)
        self._gt_points = int(gt_points)
        self._input_points = int(input_points)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        input_pcd = np.load(os.path.join(self.input_path, file_name))
        gt_pcd = np.load(os.path.join(self.gt_path, file_name))
        input_pcd = self._resample(input_pcd, self._input_points)
        gt_pcd = self._resample(gt_pcd, self._gt_points)
        return torch.from_numpy(input_pcd).float(), torch.from_numpy(gt_pcd).float()


class PCNRotAugCompletionDataset(CompletionDataset):
    def __init__(
        self, root_dir, split="train", input_points=2048, gt_points=16384,
        *, rot_aug: bool = True, rot_mode: str = "so3",
    ):
        super().__init__(root_dir, split=split, input_points=input_points, gt_points=gt_points)
        self._rot_aug = bool(rot_aug) and (split == "train")
        self._rot_mode = rot_mode
        if rot_mode not in ("so3", "yaw", "signflip"):
            raise ValueError("rot_mode 仅支持 so3、yaw 或 signflip")

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        input_pcd = np.load(os.path.join(self.input_path, file_name))
        gt_pcd = np.load(os.path.join(self.gt_path, file_name))
        if self._rot_aug:
            r = sample_rotation_matrix(self._rot_mode)
            input_pcd = (input_pcd.astype(np.float32) @ r.T).astype(np.float32)
            gt_pcd = (gt_pcd.astype(np.float32) @ r.T).astype(np.float32)
        input_pcd = self._resample(input_pcd, self._input_points)
        gt_pcd = self._resample(gt_pcd, self._gt_points)
        return torch.from_numpy(input_pcd).float(), torch.from_numpy(gt_pcd).float()
