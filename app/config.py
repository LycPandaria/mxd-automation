"""自动化配置：按键、阈值、检测设置。保存到 config.json。"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Config:
    # ---- 窗口 ----
    window_title: str = ""

    # ---- 检测 ----
    model_path: str = ""
    confidence: float = 0.5
    monster_classes: str = "monster"   # 逗号分隔的怪物类别名
    fps: int = 8

    # ---- 血量 ----
    hp_key: str = "f"
    hp_threshold: float = 0.5          # HP 比例低于此值则加血
    hp_region: Optional[List[int]] = None  # [x, y, w, h] 窗口内坐标
    hp_color: List[int] = field(default_factory=lambda: [255, 0, 0])  # RGB
    hp_tolerance: int = 20

    # ---- 战斗 ----
    target_key: str = "tab"            # 选中目标按键
    skills: List[dict] = field(default_factory=lambda: [
        {"name": "技能1", "key": "1", "cooldown": 1.0},
        {"name": "技能2", "key": "2", "cooldown": 3.0},
        {"name": "技能3", "key": "3", "cooldown": 8.0},
    ])
    move_to_monster: bool = False      # 检测到怪是否点击移动到怪位置

    # ---- 热键 ----
    start_stop_hotkey: str = "f12"


_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
)


def load_config() -> Config:
    """从 config.json 加载配置，文件不存在则返回默认。"""
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            valid = {k: v for k, v in data.items()
                     if k in Config.__dataclass_fields__}
            return Config(**valid)
        except Exception:
            pass
    return Config()


def save_config(cfg: Config):
    """保存配置到 config.json。"""
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def config_path() -> str:
    return _CONFIG_PATH
