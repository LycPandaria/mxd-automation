"""配置加载器。

加载策略（双层覆盖）：
  1. 先读 ``config/default.yaml`` 作为默认值（程序自带，请勿手动改）。
  2. 再读 ``config/user.json`` 覆盖同名字段（运行时用户改动写入此文件）。

对外保持扁平字段名（``cfg.hp_key`` / ``cfg.hp_threshold`` 等），与旧版
``app/config.py`` 兼容，UI 层迁移成本最小。YAML/JSON 内部使用嵌套结构
(``hp.key`` / ``hp.threshold``)，加载/保存时做扁平 ↔ 嵌套的双向转换。

需要 PyYAML：``pip install pyyaml``。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# 路径常量
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_YAML = os.path.join(_BASE_DIR, "config", "default.yaml")
_USER_JSON = os.path.join(_BASE_DIR, "config", "user.json")


@dataclass
class Config:
    """运行时配置（字段与旧版 app/config.py 兼容）。

    字段保持扁平命名，加载时从嵌套 YAML/JSON 转换而来。
    """

    # ---- 窗口 ----
    window_title: str = ""

    # ---- 检测 ----
    model_path: str = ""
    confidence: float = 0.5
    monster_classes: str = "monster"
    fps: int = 13

    # ---- 血量 ----
    hp_key: str = "f"
    hp_threshold: float = 0.99
    hp_region: Optional[List[int]] = None
    hp_color: List[int] = field(default_factory=lambda: [231, 34, 34])
    hp_tolerance: int = 20

    # ---- 蓝量 ----
    mp_key: str = "g"
    mp_threshold: float = 0.3
    mp_region: Optional[List[int]] = None
    mp_color: List[int] = field(default_factory=lambda: [3, 130, 232])
    mp_tolerance: int = 20

    # ---- 战斗 ----
    target_key: str = "tab"
    skills: List[dict] = field(default_factory=lambda: [
        {"name": "技能1", "key": "1", "cooldown": 1.0},
        {"name": "技能2", "key": "2", "cooldown": 3.0},
        {"name": "技能3", "key": "3", "cooldown": 8.0},
    ])
    move_to_monster: bool = False

    # ---- 热键 ----
    start_stop_hotkey: str = "f12"

    # ---- 人类模拟参数（新架构预留，UI 暂不暴露）----
    humanize_key_delay_min: float = 0.02
    humanize_key_delay_max: float = 0.06
    humanize_press_cooldown: float = 1.5


# ---------------- 扁平 ↔ 嵌套 转换 ----------------

def _flat_to_nested(cfg: Config) -> dict:
    """扁平字段 → 嵌套字典（用于写 user.json）。"""
    return {
        "window": {"title": cfg.window_title},
        "detection": {
            "model_path": cfg.model_path,
            "confidence": cfg.confidence,
            "monster_classes": cfg.monster_classes,
            "fps": cfg.fps,
        },
        "hp": {
            "key": cfg.hp_key,
            "threshold": cfg.hp_threshold,
            "color": list(cfg.hp_color) if cfg.hp_color else [],
            "tolerance": cfg.hp_tolerance,
            "region": list(cfg.hp_region) if cfg.hp_region else [],
        },
        "mp": {
            "key": cfg.mp_key,
            "threshold": cfg.mp_threshold,
            "color": list(cfg.mp_color) if cfg.mp_color else [],
            "tolerance": cfg.mp_tolerance,
            "region": list(cfg.mp_region) if cfg.mp_region else [],
        },
        "combat": {
            "target_key": cfg.target_key,
            "move_to_monster": cfg.move_to_monster,
            "skills": [dict(s) for s in cfg.skills],
        },
        "hotkey": {"start_stop": cfg.start_stop_hotkey},
        "humanize": {
            "key_delay_min": cfg.humanize_key_delay_min,
            "key_delay_max": cfg.humanize_key_delay_max,
            "press_cooldown": cfg.humanize_press_cooldown,
        },
    }


def _nested_to_flat(data: dict, cfg: Config) -> Config:
    """嵌套字典 → 扁平字段（赋值到现有 Config 实例）。"""
    win = data.get("window", {})
    cfg.window_title = win.get("title", cfg.window_title)

    det = data.get("detection", {})
    cfg.model_path = det.get("model_path", cfg.model_path)
    cfg.confidence = float(det.get("confidence", cfg.confidence))
    cfg.monster_classes = det.get("monster_classes", cfg.monster_classes)
    cfg.fps = int(det.get("fps", cfg.fps))

    hp = data.get("hp", {})
    cfg.hp_key = hp.get("key", cfg.hp_key)
    cfg.hp_threshold = float(hp.get("threshold", cfg.hp_threshold))
    hp_color = hp.get("color")
    if hp_color:
        cfg.hp_color = list(hp_color)
    cfg.hp_tolerance = int(hp.get("tolerance", cfg.hp_tolerance))
    hp_region = hp.get("region")
    if hp_region:
        cfg.hp_region = list(hp_region)

    mp = data.get("mp", {})
    cfg.mp_key = mp.get("key", cfg.mp_key)
    cfg.mp_threshold = float(mp.get("threshold", cfg.mp_threshold))
    mp_color = mp.get("color")
    if mp_color:
        cfg.mp_color = list(mp_color)
    cfg.mp_tolerance = int(mp.get("tolerance", cfg.mp_tolerance))
    mp_region = mp.get("region")
    if mp_region:
        cfg.mp_region = list(mp_region)

    cb = data.get("combat", {})
    cfg.target_key = cb.get("target_key", cfg.target_key)
    cfg.move_to_monster = bool(cb.get("move_to_monster", cfg.move_to_monster))
    skills = cb.get("skills")
    if skills:
        cfg.skills = list(skills)

    hk = data.get("hotkey", {})
    cfg.start_stop_hotkey = hk.get("start_stop", cfg.start_stop_hotkey)

    hz = data.get("humanize", {})
    cfg.humanize_key_delay_min = float(hz.get("key_delay_min", cfg.humanize_key_delay_min))
    cfg.humanize_key_delay_max = float(hz.get("key_delay_max", cfg.humanize_key_delay_max))
    cfg.humanize_press_cooldown = float(hz.get("press_cooldown", cfg.humanize_press_cooldown))

    return cfg


# ---------------- 对外 API ----------------

def load_config() -> Config:
    """加载配置：先读 default.yaml，再用 user.json 覆盖。"""
    cfg = Config()

    # 1. default.yaml
    if _YAML_AVAILABLE and os.path.exists(_DEFAULT_YAML):
        try:
            with open(_DEFAULT_YAML, "r", encoding="utf-8") as f:
                default_data = yaml.safe_load(f) or {}
            cfg = _nested_to_flat(default_data, cfg)
        except Exception:
            pass

    # 2. user.json 覆盖
    if os.path.exists(_USER_JSON):
        try:
            with open(_USER_JSON, "r", encoding="utf-8") as f:
                user_data = json.load(f)
            cfg = _nested_to_flat(user_data, cfg)
        except Exception:
            pass

    return cfg


def save_user_config(cfg: Config):
    """保存用户配置到 user.json（不写 default.yaml）。"""
    os.makedirs(os.path.dirname(_USER_JSON), exist_ok=True)
    data = _flat_to_nested(cfg)
    with open(_USER_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 向后兼容：旧代码用 save_config / config_path 名称
save_config = save_user_config


def default_yaml_path() -> str:
    return _DEFAULT_YAML


def user_json_path() -> str:
    return _USER_JSON


# 旧代码兼容：config_path() 返回 user.json 路径
def config_path() -> str:
    return _USER_JSON
