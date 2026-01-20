@echo off
REM Script pentru pornirea aplicației web Gradio
REM Limbaje Formale - UTCN

echo ========================================
echo 🚀 PORNIRE APLICATIE WEB SUMMARIZATION
echo ========================================
echo.

REM Verifică dacă mediul virtual există
if not exist "venv\Scripts\activate.bat" (
    echo [EROARE] Mediul virtual nu exista!
    echo Ruleaza mai intai: setup_venv.bat
    echo.
    pause
    exit /b 1
)

REM Activează mediul virtual
echo [1/2] Activare mediu virtual...
call venv\Scripts\activate.bat

REM Verifică dacă gradio e instalat
python -c "import gradio" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Gradio nu este instalat. Se instaleaza...
    pip install gradio>=4.0.0
)

echo [2/2] Pornire aplicatie web...
echo.
echo ========================================
echo ✅ Aplicatia va porni in browser!
echo ========================================
echo.
echo 📱 URL Local: http://127.0.0.1:7860
echo.
echo Pentru a opri aplicatia: apasa CTRL+C
echo.
echo ========================================
echo.

REM Rulează aplicația
python app.py

pause
