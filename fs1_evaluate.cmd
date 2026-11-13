@echo off
"%~dp0.venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.optimization.evaluation --official-maps Map06 Map07t4 Map08 Map09 --controllers advanced2 advanced2_fs1_area advanced2_fs1_gain advanced5_narrow --blue-variant adapted --rounds 1 --seed 742 --output "%~dp0output/semifinal_fs1_20261113/pilot" %*
if errorlevel 1 exit /b %errorlevel%
if not exist "%~dp0output/semifinal_fs1_20261113/pilot/results.json" exit /b 0
"%~dp0.venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.optimization.evaluation.fs1_review "%~dp0output/semifinal_fs1_20261113/pilot"
exit /b %errorlevel%
