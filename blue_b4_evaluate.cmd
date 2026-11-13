@echo off
call "%~dp0blue_evaluate.cmd" --output "%~dp0output\sf13\b4_pilot" --variants adapted semantic_search --rounds 2 --lifecycle %*
if errorlevel 1 exit /b %errorlevel%
if not exist "%~dp0output\sf13\b4_pilot\adapted\results.json" exit /b 0
"%~dp0.venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.blue_lifecycle_report "%~dp0output\sf13\b4_pilot" --output "%~dp0output\sf13\b4_pilot\report"
exit /b %errorlevel%
