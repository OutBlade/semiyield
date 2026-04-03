@echo off
title SemiYield Intelligence Platform

set CF=C:\Users\Sebastian\AppData\Local\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe

echo Starting SemiYield dashboard...
start "Streamlit" /min cmd /c "cd /d C:\Users\Sebastian\SemiYield && python -m streamlit run dashboard\app.py --server.port 8501 --server.headless true"

timeout /t 5 /nobreak > nul

echo Starting Cloudflare Tunnel...
start "Cloudflare" /min cmd /c "%CF% tunnel --url http://localhost:8501 --no-autoupdate 2> C:\Users\Sebastian\SemiYield\cloudflare.log"

timeout /t 8 /nobreak > nul

echo.
echo Dashboard running.
echo.
echo Local:   http://localhost:8501
echo.
echo Fetching public URL...
for /f "delims=" %%i in ('python -c "import re,open; log=open('C:/Users/Sebastian/SemiYield/cloudflare.log').read(); m=re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', log); print(m.group(0) if m else 'URL not ready yet')"') do set URL=%%i
echo Public:  %URL%
echo.
echo Press any key to open the dashboard in your browser...
pause > nul
start http://localhost:8501
