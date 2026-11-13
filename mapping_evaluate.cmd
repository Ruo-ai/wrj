@echo off
"%~dp0.venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.optimization.evaluation.red_mapping_batch %*
exit /b %errorlevel%
