@echo off
"%~dp0.venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.optimization.evaluation.blue_transport_batch %*
exit /b %errorlevel%
