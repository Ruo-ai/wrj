@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.red_p2l17_batch --maps Map08 Map09 --controllers rl11_lines rl17_repair --rounds 1 --run
exit /b %errorlevel%
