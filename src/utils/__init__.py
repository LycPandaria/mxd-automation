"""工具类。

模块：
  - logger:        日志系统（基于标准库 logging，可选 UI 回调）
  - config_loader: YAML/JSON 配置加载
  - geometry:      几何计算（距离、范围判断）
"""
from .logger import get_logger  # noqa: F401
from .config_loader import (  # noqa: F401
    Config, load_config, save_user_config, save_config,
    config_path,
)
from .geometry import distance, in_range  # noqa: F401