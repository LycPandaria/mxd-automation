@echo off
setlocal
title MXD 游戏辅助

rem ==== 管理员权限自检 ====
rem 游戏客户端常以管理员身份运行：普通权限启动的辅助程序，按键会被 Windows
rem UIPI 机制静默拦截（模型识别正常但角色一动不动）。这里自动弹 UAC 提权重启。
fltmc >nul 2>&1
if not errorlevel 1 goto :elevated

echo 检测到当前不是管理员权限，正在弹出 UAC 提权窗口，请在弹窗中点"是"...
powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
if not errorlevel 1 exit /b 0

echo.
echo [WARN] 未获得管理员权限（UAC 被取消），将以普通权限继续运行。
echo        若游戏以管理员身份运行，按键将无法生效！
echo.
pause

:elevated
rem 切换到脚本所在目录（双击运行时当前目录可能不是项目根目录）
cd /d "%~dp0"

if not exist "%~dp0main.py" (
    echo [ERROR] 找不到 main.py，请确认 start.bat 与 main.py 在同一目录。
    pause
    exit /b 1
)

rem 优先使用项目自带的虚拟环境，避免依赖版本不对（如 onnxruntime 1.20.1）
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [WARN] 未找到 .venv 虚拟环境，改用系统 Python。
    set "PYTHON=python"
)

echo 正在启动 MXD 游戏辅助...
"%PYTHON%" "%~dp0main.py"
set "EXITCODE=%errorlevel%"

rem 正常关闭窗口时退出码为 0，不打扰；异常退出时保留窗口方便看错误
if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] 程序异常退出（退出码 %EXITCODE%），请查看上方错误信息。
    pause
)

endlocal & exit /b %EXITCODE%
