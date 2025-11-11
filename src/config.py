# coding: utf-8
"""
Dataset config loader.
Only English comments and messages.
"""
from __future__ import annotations
import os, yaml
from typing import Dict, Any

DEFAULT_CFG_PATH = "configs/datasets.yml"

class DatasetCfg(dict):
    @property
    def video_id(self) -> str:
        # infer from frames_dir last component if not explicitly provided
        if "video_id" in self:
            return str(self["video_id"])
        frames = str(self.get("frames_dir", ""))
        return os.path.basename(frames) if frames else "unknown"

def load_datasets_cfg(path: str = DEFAULT_CFG_PATH) -> Dict[str, DatasetCfg]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    ds = {}
    for name, item in (data.get("datasets") or {}).items():
        ds[name] = DatasetCfg(item)
    return ds

def get_dataset(name: str, path: str = DEFAULT_CFG_PATH) -> DatasetCfg:
    all_cfg = load_datasets_cfg(path)
    if name not in all_cfg:
        raise KeyError(f"Dataset not found: {name}")
    return all_cfg[name]
