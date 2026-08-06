"""MXD 游戏辅助控制台 - 程序入口。

运行: python main.py
"""
import sys

from PyQt5.QtWidgets import QApplication

from app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MXD 游戏辅助")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
