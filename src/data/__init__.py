"""轻量入口：仅 naming（无 torch）。Dataset / 训练相关请 `from src.data.pcn_dataset import ...`。"""
from src.data.naming import PCN_TAXONOMY, TAXONOMY_ID_TO_SLUG, sample_stem

__all__ = [
    "PCN_TAXONOMY",
    "TAXONOMY_ID_TO_SLUG",
    "sample_stem",
]
