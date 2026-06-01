@echo off
echo ========================================
echo Multi-Disease Prediction System
echo FastAPI Server - 6 Disease Models
echo ========================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Starting server on http://127.0.0.1:5000
echo.
echo Available Models:
echo   - Diabetes (Tabular)
echo   - Heart Disease (Tabular)
echo   - Kidney Disease (Tabular)
echo   - Breast Cancer (Tabular)
echo   - Liver Disease (Tabular)
echo   - Pneumonia (Image-based CNN)
echo.
echo Press Ctrl+C to stop the server
echo.
uvicorn app:app --host 127.0.0.1 --port 5000
