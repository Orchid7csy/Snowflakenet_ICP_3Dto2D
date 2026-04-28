"""向后兼容：请优先使用 `src.data.pcn_dataset`。"""
from src.data.pcn_dataset import (
    CompletionDataset,
    PCNRotAugCompletionDataset,
    SnowflakeDataset,
    sample_rotation_matrix,
)

__all__ = [
    "SnowflakeDataset",
    "CompletionDataset",
    "PCNRotAugCompletionDataset",
    "sample_rotation_matrix",
]
