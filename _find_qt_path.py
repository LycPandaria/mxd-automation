import sys, os
# 先确认 PyQt5 能正常 import
from PyQt5.QtCore import QLibraryInfo, QCoreApplication
app = QCoreApplication(sys.argv)

# 打印各种 Qt 路径
print('QLibraryInfo PrefixPath:', QLibraryInfo.location(QLibraryInfo.PrefixPath))
print('QLibraryInfo PluginsPath:', QLibraryInfo.location(QLibraryInfo.PluginsPath))
print('QLibraryInfo LibrariesPath:', QLibraryInfo.location(QLibraryInfo.LibrariesPath))
print('QLibraryInfo BinariesPath:', QLibraryInfo.location(QLibraryInfo.BinariesPath))
print('QLibraryInfo DataPath:', QLibraryInfo.location(QLibraryInfo.DataPath))

# 查看 Qt5 DLL 实际加载位置（通过模块文件路径推断）
import PyQt5.QtCore as qc
print('QtCore __file__:', qc.__file__)

import ctypes, ctypes.wintypes
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi
hMod = ctypes.wintypes.HMODULE()
cbNeeded = ctypes.wintypes.DWORD()

# 枚举当前进程加载的 Qt5 DLL
for pid in [os.getpid()]:
    hProcess = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not hProcess:
        continue
    if psapi.EnumProcessModules(hProcess, ctypes.byref(hMod), ctypes.sizeof(hMod), ctypes.byref(cbNeeded)):
        n = int(cbNeeded.value / ctypes.sizeof(hMod))
        arr = (ctypes.wintypes.HMODULE * n)()
        if psapi.EnumProcessModules(hProcess, arr, cbNeeded.value, ctypes.byref(cbNeeded)):
            for i in range(n):
                name = ctypes.create_unicode_buffer(260)
                psapi.GetModuleFileNameExW(hProcess, arr[i], name, 260)
                if 'Qt5' in name.value:
                    print('LOADED:', name.value)
    kernel32.CloseHandle(hProcess)
