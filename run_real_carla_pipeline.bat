@echo off
echo ===================================================
echo CARLA Live Simulator Optuna Search Pipeline - Windows
echo ===================================================
echo.
echo [1/3] Checking/Installing standard dependencies...
pip install -r requirements.txt
echo.
echo [2/3] Checking CARLA Python API installation...
python -c "import carla" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 'carla' Python module not found!
    echo Please install the carla wheel/egg matching your simulator version.
    echo Example: pip install carla==0.9.15
    echo.
    pause
    exit /b 1
) else (
    echo [OK] 'carla' library is installed.
)
echo.
echo [3/3] Running Live CARLA Optuna Optimizer...
echo Please ensure that the CARLA Simulator is running in the background.
echo.
python carla_optuna_optimizer.py

echo.
echo ===================================================
echo Optimization Finished! Check the 'results/' folder.
echo ===================================================
pause
