"""配置加载与配置实体。

================================================================================
配置系统设计
================================================================================

  配置分为两层:
    - 默认配置: 内置在代码中，提供合理的出厂默认值
    - 用户配置: 保存在 config/user.json，覆盖默认值

  双层覆盖机制:
    1. 先加载默认配置（_defaults()）
    2. 再读取 user.json，用用户配置覆盖同名字段
    3. 最终 Config 对象包含合并后的值

  这样用户只需要配置自己关心的字段，其余使用默认值即可。

================================================================================
坐标自适应
================================================================================

  问题: 配置文件中的 HP 区域 / 自身偏移 等坐标是参考 1366×768 分辨率
       记录的，但实际运行时窗口可能不同（如 1920×1080）。

  解决: scale_region() 和 scale_offset() 根据参考分辨率与当前帧的比例，
       自动缩放坐标值。

  公式:
    scale_x = 当前帧宽 / 参考帧宽
    scale_y = 当前帧高 / 参考帧高
    scaled_x = x * scale_x
    scaled_y = y * scale_y

================================================================================
JSON 配置字段说明
================================================================================

  window_title:     游戏窗口标题（用于 FindWindow 锁定）
  reference_width:  参考分辨率宽度（坐标记录时的分辨率）
  reference_height: 参考分辨率高度
  fps:              目标帧率
  confidence:       YOLO 检测置信度阈值 (0.0~1.0)
  model_path:       YOLO 模型文件路径
  monster_classes:  怪物类别名（逗号分隔）
  floor_classes:    地板类别名
  rope_classes:    绳索类别名
  self_name:        自身角色名字（用于 OCR 定位）
  self_offset:      自身 HP 条底部到脚底的偏移像素数
  hp_region:        HP 条参考区域 [x, y, w, h]
  hp_color:         HP 条颜色 [R, G, B]
  hp_tolerance:     HP 条颜色容差
  hp_threshold:     HP 加血阈值 (0.0~1.0)
  hp_key:           加血键
  mp_region:        MP 条参考区域
  mp_color:         MP 条颜色
  mp_tolerance:     MP 条颜色容差
  mp_threshold:     MP 加蓝阈值
  mp_key:           加蓝键
  target_key:       选目标键
  skills:           技能列表 [{name, key, cooldown}, ...]
"""
import json
import os
from typing import List, Optional, Dict, Any, Tuple


# ---- 路径常量 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "user.json")

# 别名，供外部导入
config_path = DEFAULT_CONFIG_PATH
user_json_path = DEFAULT_CONFIG_PATH
default_yaml_path = os.path.join(PROJECT_ROOT, "assets", "default.yaml")


def _defaults() -> Dict[str, Any]:
    """返回默认配置字典。

    这里定义了所有配置项的默认值。用户 JSON 中未配置的字段
    会使用这些默认值。

    修改这些默认值会影响所有用户（除非用户在 user.json 中覆盖）。

    Returns:
        包含所有默认配置的字典
    """
    return {
        # ---- 窗口 ----
        "window_title": "",
        # ---- 热键 ----
        "start_stop_hotkey": "F6",
        # ---- 分辨率自适应 ----
        # 参考分辨率：坐标录制时的分辨率，所有坐标配置都基于此分辨率
        "reference_width": 1366,
        "reference_height": 768,
        # ---- 性能 ----
        "fps": 13,
        # ---- 模型 ----
        "confidence": 0.5,
        "model_path": "assets/models/best.pt",
        # ---- 类别 ----
        "monster_classes": "monster",
        "floor_classes": "floor",
        "rope_classes": "rope",
        # ---- 自身定位 ----
        "self_name": "",  # 角色脚底名字，用于 OCR 定位
        "self_offset": 85,  # HP 条底部 → 脚底的偏移像素
        # ---- HP 检测 ----
        "hp_region": None,  # [x, y, w, h] 参考分辨率下的 HP 条区域
        "hp_color": [51, 204, 51],  # HP 条绿色 RGB
        "hp_tolerance": 30,  # 颜色容差
        "hp_threshold": 0.3,  # 低于 30% 时加血
        "hp_key": "f",  # 加血快捷键
        # ---- MP 检测 ----
        "mp_region": None,
        "mp_color": [51, 153, 255],  # MP 条蓝色 RGB
        "mp_tolerance": 30,
        "mp_threshold": 0.3,
        "mp_key": "g",  # 加蓝快捷键
        # ---- 战斗 ----
        "target_key": "tab",  # 选目标键
        "skills": [
            {"name": "技能1", "key": "1", "cooldown": 1.0},
            {"name": "技能2", "key": "2", "cooldown": 3.0},
            {"name": "技能3", "key": "3", "cooldown": 8.0},
        ],
    }


class Config:
    """配置实体类。

    封装双层配置（默认 + 用户），提供属性访问和坐标缩放功能。

    用法:
        cfg = Config.load()  # 从 config/user.json 加载
        cfg = Config(overrides={"hp_region": [100, 200, 50, 10]})  # 覆盖某项

    Attributes:
        所有 JSON 配置字段都作为属性直接访问，如 cfg.hp_threshold, cfg.fps 等。
        属性名与 JSON 字段名一致（下划线命名）。
    """

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        """构造配置对象。

        Args:
            overrides: 覆盖项字典，键值对会覆盖默认配置
        """
        # 1. 加载默认配置
        data = _defaults()

        # 2. 用用户 JSON 覆盖
        if os.path.isfile(DEFAULT_CONFIG_PATH):
            try:
                with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                    user = json.load(f)
                data.update(user)  # 用户配置覆盖默认配置
            except Exception:
                pass  # JSON 解析失败时忽略，使用默认值

        # 3. 用运行时覆盖项覆盖
        if overrides:
            data.update(overrides)

        self._data = data

    # =========================================================================
    # 工厂方法
    # =========================================================================

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """从指定路径加载配置。

        Args:
            path: JSON 配置文件路径，None 时使用默认路径 config/user.json

        Returns:
            Config 实例
        """
        if path is None:
            path = DEFAULT_CONFIG_PATH
        data = _defaults()
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data.update(json.load(f))
        return cls(overrides=None)  # 上面的 data 处理逻辑在 __init__ 中

    def save(self, path: Optional[str] = None):
        """保存当前配置到 JSON 文件。

        Args:
            path: 保存路径，None 时使用默认路径
        """
        if path is None:
            path = DEFAULT_CONFIG_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def merge(self, updates: Dict[str, Any]):
        """合并配置项（运行时修改）。

        Args:
            updates: 要更新的键值对
        """
        self._data.update(updates)

    # =========================================================================
    # 属性访问：让配置项像属性一样访问
    # 例如: cfg.hp_threshold 等价于 cfg._data["hp_threshold"]
    # =========================================================================

    def __getattr__(self, name: str):
        """属性访问降级到 _data 字典。

        如果正常属性找不到（如 __dict__ 中没有），
        就在 _data 字典中查找。这样可以让配置项像属性一样访问。

        例如: cfg.hp_threshold → cfg._data["hp_threshold"]
        """
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config 没有 '{name}' 字段")

    def __setattr__(self, name: str, value):
        """属性赋值：保存在 _data 中。

        非私有属性（不以下划线开头）直接写入 _data 字典。
        私有属性（如 _data）正常走标准属性赋值。
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    # =========================================================================
    # 坐标自适应缩放
    # =========================================================================

    def scale_region(self, region: Optional[List],
                     frame_width: int, frame_height: int) -> Optional[Tuple[int, int, int, int]]:
        """将配置区域转换为当前帧像素坐标。

        支持两种格式:
          1. 百分比 [x%, y%, w%, h%]  每个值 0.0~1.0，相对帧宽高
          2. 像素 [x, y, w, h]  整数，基于 reference_width/reference_height 缩放

        Args:
            region:       区域 [x, y, w, h]，None 返回 None
            frame_width:  当前帧宽度（像素）
            frame_height: 当前帧高度（像素）

        Returns:
            区域 (x, y, w, h) 像素坐标，无效时返回 None
        """
        if region is None:
            return None
        if len(region) < 4:
            return None

        # 判断格式：浮点或 <= 1.0 → 百分比，否则 → 像素缩放
        is_percent = any(isinstance(v, float) for v in region) or max(region) <= 1.0

        if is_percent:
            # 百分比：直接乘以帧尺寸
            sx_i = int(region[0] * frame_width)
            sy_i = int(region[1] * frame_height)
            sw_i = int(region[2] * frame_width)
            sh_i = int(region[3] * frame_height)
        else:
            # 像素：按参考分辨率缩放
            sx = frame_width / self.reference_width
            sy = frame_height / self.reference_height
            sx_i = int(region[0] * sx)
            sy_i = int(region[1] * sy)
            sw_i = int(region[2] * sx)
            sh_i = int(region[3] * sy)

        # 边界保护
        if sx_i < 0:
            sw_i += sx_i
            sx_i = 0
        if sy_i < 0:
            sh_i += sy_i
            sy_i = 0
        if sx_i + sw_i > frame_width:
            sw_i = frame_width - sx_i
        if sy_i + sh_i > frame_height:
            sh_i = frame_height - sy_i
        if sw_i <= 0 or sh_i <= 0:
            return None
        return (sx_i, sy_i, sw_i, sh_i)

    def scale_offset(self, offset: int, frame_height: int) -> int:
        """将参考分辨率下的偏移量缩放到当前帧高度。

        公式: scaled_offset = offset * (frame_height / reference_height)

        Args:
            offset:       参考分辨率下的偏移（像素）
            frame_height: 当前帧高度（像素）

        Returns:
            缩放后的偏移量
        """
        return int(offset * frame_height / self.reference_height)


def load_config(path: Optional[str] = None) -> Config:
    """快捷函数：加载配置。

    Args:
        path: JSON 配置文件路径，None 使用默认路径

    Returns:
        Config 实例
    """
    return Config.load(path)


def save_config(config: Config, path: str):
    """保存配置到指定路径。

    Args:
        config: Config 实例
        path: 保存路径
    """
    config.save(path)


def save_user_config(config: Config):
    """保存配置到默认用户配置文件。

    Args:
        config: Config 实例
    """
    config.save(DEFAULT_CONFIG_PATH)