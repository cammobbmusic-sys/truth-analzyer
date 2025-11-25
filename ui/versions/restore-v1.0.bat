@echo off
REM Restore UI to Version 1.0 Prototype

echo 🔄 Restoring UI to Version 1.0 Prototype...

REM Restore the template
copy index-v1.0.html ..\templates\index.html

REM Restore the dashboard
copy dashboard-v1.0.py ..\dashboard.py

echo ✅ Version 1.0 restored successfully!
echo Restart the Flask server to see the changes.
pause
