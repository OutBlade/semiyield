@echo off
title SemiYield Intelligence Platform

echo Starting SemiYield dashboard...
start "Streamlit" /min cmd /c "cd /d C:\Users\Sebastian\SemiYield && python -m streamlit run dashboard\app.py --server.port 8501 --server.headless true"

timeout /t 5 /nobreak > nul

echo Starting ngrok tunnel...
start "ngrok" /min cmd /c "C:\Users\Sebastian\ngrok-bin\ngrok.exe http 8501 --domain=unstubbled-harmonizable-siu.ngrok-free.dev"

timeout /t 5 /nobreak > nul

echo.
echo Dashboard running.
echo.
echo Local:   http://localhost:8501
echo Public:  https://unstubbled-harmonizable-siu.ngrok-free.dev
echo.
echo Press any key to open the dashboard in your browser...
pause > nul
start http://localhost:8501
