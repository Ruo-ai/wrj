@echo off
"%~dp0.venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.optimization.evaluation.fs0 %*
exit /b %errorlevel%
