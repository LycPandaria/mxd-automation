"""MXD 游戏辅助控制台 - 程序入口（GUI 模式）。

================================================================================
启动方式
================================================================================

  GUI 模式（推荐）:
    python main.py
    启动 PyQt5 图形界面，可以锁定窗口、配置参数、查看实时预览。

  CLI 模式（无 GUI）:
    python -m src.main
    纯命令行模式，适合调试或服务器运行。

================================================================================
sys.path 处理
================================================================================

  项目根目录被加入 sys.path，这样 `src` 和 `ui` 可以作为顶层包导入:
    from ui.main_window import MainWindow
    from src.perception.yolo_detector import Detector

  这避免了相对导入的路径问题。
"""
import sys
import os

# 把项目根目录加入 sys.path，让 `src` / `ui` 作为顶层包可被导入
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
    """启动 PyQt5 GUI 主窗口。

    流程:
      1. 创建 QApplication（PyQt5 应用实例）
      2. 创建 MainWindow（主窗口，包含所有 UI 控件）
      3. 显示窗口
      4. 进入事件循环（app.exec_()），直到用户关闭窗口
    """
    app = QApplication(sys.argv)
    app.setApplicationName("MXD 游戏辅助")
    window = MainWindow()
    window.show()
    # app.exec_() 启动 Qt 事件循环，阻塞直到窗口关闭
    # 返回值是退出码，传给 sys.exit
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()