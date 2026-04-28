"""
与 Snet `core/datasets.utils.UpSamplePoints` 一致：FPS(512~1024) 再补点到 n_points（含重复）。

需在 import 前将项目根与 Snet 根加入 sys.path。
"""

from __future__ import annotations

import random

import numpy as np


def farthest_point_sample(point: np.ndarray, npoint: int) -> np.ndarray:
    """与 Snet completion/core/datasets/utils.farthest_point_sample 一致。"""
    n, _d = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((n,)) * 1e10
    farthest = np.random.randint(0, n)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return point[centroids.astype(np.int32)]


class UpSamplePointsSnet:
    """与 `core.datasets.utils.UpSamplePoints` 一致，避免从 utils 拉入 h5py/cv2 等依赖。"""

    def __init__(self, parameters: dict):
        self.n_points = int(parameters["n_points"])

    def __call__(self, ptcloud: np.ndarray) -> np.ndarray:
        n_valid = random.randint(512, 1024)
        ptcloud = farthest_point_sample(ptcloud, n_valid)
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        return np.concatenate((ptcloud, ptcloud[choice]))
