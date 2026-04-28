"""
PCN：ShapeNet synset → 论文类名、文件名片段（小写 slug）。"""
from __future__ import annotations

# (taxonomy_id, 英文类名, 文件名用短 slug，便于阅读)
PCN_TAXONOMY: list[tuple[str, str, str]] = [
    ("02691156", "Plane", "plane"),
    ("02933112", "Cabinet", "cabinet"),
    ("02958343", "Car", "car"),
    ("03001627", "Chair", "chair"),
    ("03636649", "Lamp", "lamp"),
    ("04256520", "Couch", "couch"),
    ("04379243", "Table", "table"),
    ("04530566", "Boat", "boat"),
]

TAXONOMY_ID_TO_SLUG: dict[str, str] = {t[0]: t[2] for t in PCN_TAXONOMY}


def sample_stem(*, split: str, synset: str, model_id: str, view: str | int) -> str:
    """
    有语义、且与 SnowflakeDataset 兼容：input/ 与 gt/ **同名** .npy。
    例: pcn__test__plane_02691156__<model_id>__view_00
    """
    slug = TAXONOMY_ID_TO_SLUG.get(synset, f"cat_{synset}")
    if isinstance(view, int):
        v = f"view_{view:02d}"
    else:
        v = f"view_{view}" if not str(view).startswith("view_") else str(view)
    return f"pcn__{split}__{slug}_{synset}__{model_id}__{v}"
