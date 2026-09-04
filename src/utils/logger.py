"""日志系统。

================================================================================
设计
================================================================================

  基于 Python 标准库 logging，配置控制台输出 + 文件持久化。
  全局单例模式：整个进程只有一个 logger 实例。

================================================================================
输出目标
================================================================================

  1. 控制台 (stdout)：实时查看
  2. 文件：logs/mxd.log（按天轮转，保留 7 天）

================================================================================
输出格式
================================================================================

  控制台: [HH:MM:SS] [LEVEL] 消息内容
  文件:   [YYYY-MM-DD HH:MM:SS] [LEVEL] 消息内容

  例如:
    [14:32:05] [INFO] 已锁定窗口: 冒险岛
    [14:32:06] [ERROR] 截图失败: window not found

================================================================================
使用方式
================================================================================

  from src.utils.logger import get_logger
  log = get_logger()
  log.info("程序启动")
  log.error("出错了")
  log.debug("调试信息")  # 需要 level=logging.DEBUG
"""
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

_LOGGER_NAME = "mxd"   # logger 名称，全局唯一
_logger = None          # 全局单例
_log_dir_cache = None   # 日志目录缓存


def _log_dir() -> str:
    """返回日志目录：<项目根>/logs（打包后为 exe 同目录/logs）。"""
    global _log_dir_cache
    if _log_dir_cache is not None:
        return _log_dir_cache
    if getattr(sys, "frozen", False):
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        # src/utils/logger.py → 向上三级到项目根目录
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _log_dir_cache = os.path.join(base, "logs")
    return _log_dir_cache


def get_logger(level: int = logging.INFO) -> logging.Logger:
    """获取全局 logger 单例。

    首次调用:
      1. 创建名为 "mxd" 的 logger
      2. 设置 propagate=False（避免根 logger 重复输出）
      3. 添加 StreamHandler 输出到 stdout（控制台）
      4. 添加 TimedRotatingFileHandler 持久化到 logs/mxd.log（按天轮转）

    后续调用:
      只更新日志级别，不重复创建 handler。

    Args:
        level: 日志级别，常用 logging.INFO / logging.DEBUG / logging.WARNING

    Returns:
        logging.Logger 实例
    """
    global _logger
    if _logger is not None:
        # 已经初始化过，只更新级别
        _logger.setLevel(level)
        return _logger

    # 首次初始化
    _logger = logging.getLogger(_LOGGER_NAME)
    _logger.setLevel(level)
    _logger.propagate = False  # 不向根 logger 传播，避免重复输出

    if _logger.handlers:
        return _logger

    # 1. 控制台输出（stdout）
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    _logger.addHandler(console)

    # 2. 文件持久化（logs/mxd.log，按天轮转，保留 7 天）
    try:
        log_dir = _log_dir()
        os.makedirs(log_dir, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            os.path.join(log_dir, "mxd.log"),
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        _logger.addHandler(file_handler)
    except Exception:
        # 文件日志失败不阻塞程序（如目录不可写），控制台仍可用
        pass

    return _logger
