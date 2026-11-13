@echo off
"%~dp0.venv\Scripts\python.exe" -B -m swarm_rescue.solutions.devtools.red_mapping_cells_batch --output "%~dp0output\sf13\ra4_pilot" --maps Map08 Map09 yh_map_fs_1 yh_map_fs_2 --controllers advanced2_mapping a4_cells --rounds 1 --full-pairs %*
exit /b %errorlevel%
