"""MXD 游戏辅助控制台 - 程序入口。

启动 PyQt5 GUI 主窗口。

运行:
    python main.py

CLI 模式（无 GUI，仅主循环 + 日志）:
    python -m src.main
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
    app = QApplication(sys.argv)
    app.setApplicationName("MXD 游戏辅助")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
