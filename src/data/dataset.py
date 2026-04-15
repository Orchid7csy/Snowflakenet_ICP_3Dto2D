import os
import torch
import numpy as np
from torch.utils.data import Dataset

class SnowflakeDataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=2048, transform=None):
        """
        root_dir: data/processed 的路径
        split: 'train' 或 'test'
        """
        self.input_path = os.path.join(root_dir, split, 'input')
        self.gt_path = os.path.join(root_dir, split, 'gt')
        self.file_list = [f for f in os.listdir(self.input_path) if f.endswith('.npy')]
        self.num_points = num_points
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.file_list)

    def _resample(self, pcd, n):
        """
        对点云进行重采样以达到固定点数 n
        """
        curr_n = pcd.shape[0]
        if curr_n == n:
            return pcd
        if curr_n > n:
            # 如果点多了（理论上你的 preprocessing 不会出现），进行随机下采样
            idx = np.random.choice(curr_n, n, replace=False)
        else:
            # 如果点少了（你的 dropout 情况），进行随机重复填充
            idx = np.concatenate([
                np.arange(curr_n), 
                np.random.choice(curr_n, n - curr_n, replace=True)
            ])
        return pcd[idx]

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        
        # 加载残缺输入和真值
        input_pcd = np.load(os.path.join(self.input_path, file_name))
        gt_pcd = np.load(os.path.join(self.gt_path, file_name))

        # 强制对齐点数到 2048
        input_pcd = self._resample(input_pcd, self.num_points)
        # GT 理论上已经是 2048，但为了保险也跑一遍
        gt_pcd = self._resample(gt_pcd, self.num_points)

        # 转换为 Tensor (BxNx3 格式要求)
        input_tensor = torch.from_numpy(input_pcd).float()
        gt_tensor = torch.from_numpy(gt_pcd).float()

        # 如果有数据增强（如旋转、缩放），在此处调用
        if self.transform:
            input_tensor, gt_tensor = self.transform(input_tensor, gt_tensor)

        return input_tensor, gt_tensor