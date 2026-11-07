@echo off
chcp 65001 >nul
setlocal
title MXD 游戏辅助

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
