@echo off
chcp 65001 >nul
REM 双击即可：拉取最新代码 + 热重载 AstrBot 插件（无需重启后端）
set PY=D:\Local\AstrBot\backend\python\python.exe
if not exist "%PY%" set PY=python
cd /d "%~dp0"
"%PY%" hot_reload.py --pull %*
echo.
pause
