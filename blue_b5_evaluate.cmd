@echo off
rem Optional diagnostic recheck: Map08 only, 4 independent variants x 2 seeds.
call "%~dp0blue_evaluate.cmd" --output "%~dp0output\sf13\b5_recheck" --maps Map08 --variants adapted semantic_search narrow_empty frontier_tasks --rounds 2 --lifecycle %*
if errorlevel 1 exit /b %errorlevel%
if not exist "%~dp0output\sf13\b5_recheck\adapted\results.json" exit /b 0
"%~dp0.venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.blue_lifecycle_report "%~dp0output\sf13\b5_recheck" --output "%~dp0output\sf13\b5_recheck\report\lifecycle"
if errorlevel 1 exit /b %errorlevel%
"%~dp0.venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.blue_narrow_report --micro "%~dp0output\sf13\b5_micro_final" --batch "%~dp0output\sf13\b5_recheck" --output "%~dp0output\sf13\b5_recheck\report"
exit /b %errorlevel%
