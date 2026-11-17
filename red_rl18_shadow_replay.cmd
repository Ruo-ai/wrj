@echo off
setlocal
cd /d "%~dp0"
rem Offline shadow replay only: Map08/09, original r1 seeds, 10 aircraft tapes.
rem No flight, no official entry change, no navigation/map write-back.
set "RL18_BASE=output/sf13/rl18_audit_20261116/stage_a_v2"
set "RL18_SOURCE=output/sf13/rl17_m89_once_20261116"
".venv\Scripts\python.exe" -B -c "from pathlib import Path; from swarm_rescue.solutions.devtools.red_recovery18_audit import read,digest; r=Path('output/sf13/rl18_audit_20261116'); b=read(r/'stage_b_v1/Map08/results.json'); assert digest('src/swarm_rescue/solutions/devtools/red_scan18_shadow.py')==b['source_sha256'],'Shadow source changed; prepare a new experiment'; docs=[read(p) for p in (r/'stage_a_v2').glob('*/results.json')]; assert len(docs)==4 and all(d['valid'] for d in docs); assert all(digest(p)==h for d in docs for p,h in d['source_code'].items()),'Baseline source changed; rerun Stage A first'; assert all(digest(Path(d['source'])/p)==h for d in docs for p,h in d['source_hashes'].items()),'Recorded inputs changed'"
if errorlevel 1 exit /b 1
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RL18_STAMP=%%i"
if not defined RL18_STAMP exit /b 1
set "RL18_OUT=output/sf13/rl18_shadow_recheck_%RL18_STAMP%"
if exist "%RL18_OUT%" exit /b 1
if exist "%RL18_OUT%_resources.json" exit /b 1
".venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.windows_pressure "%RL18_OUT%_resources.json" swarm_rescue.solutions.devtools.red_scan18_replay --batch "%RL18_SOURCE%" --baseline "%RL18_BASE%" --output "%RL18_OUT%"
if errorlevel 1 exit /b 1
echo Offline replay complete: %RL18_OUT%
endlocal
