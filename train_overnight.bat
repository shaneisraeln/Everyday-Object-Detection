@echo off
title Everyday Object Detection - Overnight Training
echo ======================================================================
echo   Launching YOLO11m Overnight Fine-Tuning Pipeline
echo ======================================================================
echo.
python run_overnight_training.py --epochs 200 --batch 8
pause
