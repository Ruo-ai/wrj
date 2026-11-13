@echo off
"%~dp0.venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.optimization.dataset_manager build %*
exit /b %errorlevel%
