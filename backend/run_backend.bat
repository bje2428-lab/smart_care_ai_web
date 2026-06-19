@echo off
echo Smart Care AI Backend Start
python -m uvicorn main:app --reload
pause
