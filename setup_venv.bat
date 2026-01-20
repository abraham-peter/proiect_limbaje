@echo off
REM Script pentru configurarea automată a mediului virtual Python
REM Limbaje Formale - UTCN

echo ========================================
echo SETUP MEDIU VIRTUAL PYTHON
echo ========================================
echo.

REM Verifică dacă Python este instalat
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [EROARE] Python nu este instalat sau nu este in PATH!
    echo Descarca Python de la: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python detectat:
python --version
echo.

REM Creează mediul virtual
echo [1/4] Creare mediu virtual (venv)...
python -m venv venv

if not exist "venv\Scripts\activate.bat" (
    echo [EROARE] Mediul virtual nu a fost creat!
    pause
    exit /b 1
)

echo [OK] Mediu virtual creat!
echo.

REM Activează mediul virtual
echo [2/4] Activare mediu virtual...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [3/4] Upgrade pip...
python -m pip install --upgrade pip

REM Instalează dependințele
echo [4/4] Instalare dependinte (transformers, torch, etc.)...
echo Aceasta poate dura 2-5 minute...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [EROARE] Instalarea dependintelor a esuat!
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ SETUP COMPLET!
echo ========================================
echo.
echo Pentru a folosi proiectul:
echo   1. Activeaza mediul virtual: venv\Scripts\activate
echo   2. Ruleaza scriptul: python summarization_bart.py
echo.
echo Pentru a dezactiva mediul virtual: deactivate
echo.
pause
