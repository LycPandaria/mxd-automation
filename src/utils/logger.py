"""日志系统。

基于标准库 ``logging``，配置控制台输出 + 可选 UI 回调（用于把日志推送到
PyQt 主窗口的日志框）。
"""
import logging
import sys

_LOGGER_NAME = "mxd"
_logger = None


def get_logger(level: int = logging.INFO) -> logging.Logger:
    """获取全局 logger 单例。

    首次调用会配置 StreamHandler 输出到 stdout。重复调用只调整 level。
    """
    global _logger
    if _logger is not None:
        _logger.setLevel(level)
        return _logger

    _logger = logging.getLogger(_LOGGER_NAME)
    _logger.setLevel(level)
    _logger.propagate = False  # 避免根 logger 重复输出

    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)

    return _logger
