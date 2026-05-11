@echo off
chcp 65001 >nul
title 多模态情感识别系统

echo ============================================
echo    🎭 多模态情感识别系统 - 一键启动
echo ============================================
echo.

:: 检查 conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] 未检测到 conda，尝试直接使用 pip...
    goto :USE_PIP
)

:: 检查 emotion 环境是否存在
conda env list | findstr /C:"emotion" >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/3] 首次运行，创建 conda 环境 (Python 3.11)...
    conda create -n emotion python=3.11 -y
    if %errorlevel% neq 0 (
        echo [!] conda 环境创建失败，尝试 pip 方式...
        goto :USE_PIP
    )
)

:: 激活环境
echo [1/3] 激活 conda 环境 emotion...
call conda activate emotion

:: 安装依赖
echo [2/3] 检查并安装依赖...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo [!] 依赖安装失败，请检查网络连接
    pause
    exit /b 1
)

goto :LAUNCH

:USE_PIP
echo [2/3] 检查并安装依赖...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo [!] 依赖安装失败，请检查网络连接
    pause
    exit /b 1
)

:LAUNCH
echo [3/3] 启动系统...
echo.
echo ============================================
echo    浏览器将自动打开 http://localhost:7860
echo    按 Ctrl+C 停止服务
echo ============================================
echo.
python frontend_app.py

pause
