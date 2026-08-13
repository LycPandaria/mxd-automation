"""日志系统。

================================================================================
设计
================================================================================

  基于 Python 标准库 logging，配置控制台输出。
  全局单例模式：整个进程只有一个 logger 实例。

================================================================================
输出格式
================================================================================

  [HH:MM:SS] [LEVEL] 消息内容

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
import sys

_LOGGER_NAME = "mxd"  # logger 名称，全局唯一
_logger = None         # 全局单例


def get_logger(level: int = logging.INFO) -> logging.Logger:
    """获取全局 logger 单例。

    首次调用:
      1. 创建名为 "mxd" 的 logger
      2. 设置 propagate=False（避免根 logger 重复输出）
      3. 添加 StreamHandler 输出到 stdout

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

    if not _logger.handlers:
        # 创建控制台输出处理器
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        # 设置格式: [时间] [级别] 消息
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)

    return _logger