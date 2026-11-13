@echo off
"%~dp0.venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.optimization.map_generator.append_maps %*
exit /b %errorlevel%
