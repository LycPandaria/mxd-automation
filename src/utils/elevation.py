"""进程权限（提权）检查 —— UIPI 拦截自检。

================================================================================
为什么要做这个检查
================================================================================

  Windows 的 UIPI（用户界面特权隔离）规定: 低权限进程注入到高权限进程的
  键盘/鼠标输入会被系统【静默丢弃】:

    - SendInput 返回"成功"、GetLastError 也不报错（微软文档明确说明
      返回值不反映 UIPI 拦截），日志里看不到任何异常；
    - PostMessageW 发送的鼠标消息同样被丢弃；
    - 但截图/模型识别不受影响。

  症状恰好是: 模型识别一切正常，角色却一动不动（指令进不去游戏）。
  游戏客户端（Maplestory_Classic.exe）常以管理员身份运行，
  辅助程序若以普通权限启动就会命中此拦截。

  修复: 用与游戏一致的（或更高）权限启动本程序 ——
  右键 start.bat → 以管理员身份运行（start.bat 已带自动提权）。

================================================================================
实现
================================================================================

  - OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) 打开目标进程
    （同用户的提权进程也允许有限查询）；
  - OpenProcessToken + GetTokenInformation(TokenElevation) 查询是否提权。

  查询失败（跨用户/进程退出等）返回 None，调用方按"未知"处理、不误报。
"""
from typing import Callable, Optional

import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
advapi32 = ctypes.windll.advapi32

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
_TOKEN_ELEVATION = 20  # TokenInformationClass.TokenElevation


def is_self_elevated() -> bool:
    """本进程是否以管理员（提权）身份运行。查询失败按未提权处理。"""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def is_process_elevated(pid: int) -> Optional[bool]:
    """查询指定进程是否以管理员（提权）身份运行。

    Args:
        pid: 进程 ID

    Returns:
        True 提权 / False 未提权 / None 查询失败（按"未知"处理）
    """
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return None
    try:
        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(handle, TOKEN_QUERY,
                                         ctypes.byref(token)):
            return None
        try:
            elevation = wintypes.DWORD(0)
            returned = wintypes.DWORD(0)
            ok = advapi32.GetTokenInformation(
                token, _TOKEN_ELEVATION, ctypes.byref(elevation),
                ctypes.sizeof(elevation), ctypes.byref(returned))
            return bool(elevation.value) if ok else None
        finally:
            kernel32.CloseHandle(token)
    finally:
        kernel32.CloseHandle(handle)


def is_window_elevated(hwnd: int) -> Optional[bool]:
    """查询窗口所属进程是否以管理员（提权）身份运行。

    Returns:
        True 提权 / False 未提权 / None 查询失败（按"未知"处理）
    """
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(int(hwnd), ctypes.byref(pid))
    if not pid.value:
        return None
    return is_process_elevated(pid.value)


def warn_uipi_mismatch(hwnd: int, title: str, on_log: Callable[[str], None]):
    """锁定游戏窗口后的权限自检：游戏提权而本程序未提权时报警。

    只在"游戏=管理员、本程序=普通权限"这种必被 UIPI 拦截的组合下警告；
    反过来（本程序管理员、游戏普通权限）不受 UIPI 限制，不警告。
    查询失败（未知）保持安静，避免误报。

    Args:
        hwnd:  已锁定的游戏窗口句柄
        title: 窗口标题（用于日志展示）
        on_log: 日志回调 (message: str) -> None
    """
    game_elevated = is_window_elevated(hwnd)
    if not game_elevated:  # None（未知）或 False（未提权）都不报警
        return
    if is_self_elevated():
        return
    on_log(
        f"[权限] 警告: 游戏「{title}」以管理员身份运行，而本程序是普通权限，"
        "按键/点击会被 Windows 静默拦截（模型识别正常但角色不动）！"
        "请关闭程序后右键 start.bat → 以管理员身份运行。"
    )
