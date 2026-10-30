@echo off
title Barca Vision Labeler - Tactical Annotation Studio
echo ========================================================
echo  Launching Barca Vision Labeler Studio...
echo ========================================================
cd /d "%~dp0"
"C:\Users\harve\AppData\Local\Programs\Python\Python311\python.exe" labeler\run_labeler.py
pause
