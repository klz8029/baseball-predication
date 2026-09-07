@echo off
REM Daily forward-test run. Registered with Windows Task Scheduler as "BaseballForwardTest".
cd /d "%~dp0"
"C:\Users\klzho\AppData\Local\Programs\Python\Python313\python.exe" forward.py >> forward_run.log 2>&1
