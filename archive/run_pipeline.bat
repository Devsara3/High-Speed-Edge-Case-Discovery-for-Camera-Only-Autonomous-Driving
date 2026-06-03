@echo off
echo ===================================================
echo CARLA Edge Case Search Pipeline - Windows
echo ===================================================

echo [1/2] Installing requirements...
pip install -r requirements.txt

echo.
echo [2/2] Running Optuna Optimizer...
python optimizer.py

echo.
echo ===================================================
echo Optimization Finished! Check the 'results/' folder.
echo ===================================================
pause
