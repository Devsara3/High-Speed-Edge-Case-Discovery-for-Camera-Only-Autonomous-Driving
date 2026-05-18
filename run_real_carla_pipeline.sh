#!/bin/bash

echo "==================================================="
echo "CARLA Live Simulator Optuna Search Pipeline - Linux"
echo "==================================================="
echo ""

echo "[1/3] Checking/Installing standard dependencies..."
pip install -r requirements.txt
echo ""

echo "[2/3] Checking CARLA Python API installation..."
python3 -c "import carla" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] 'carla' Python module not found!"
    echo "Please install the carla wheel/egg matching your simulator version."
    echo "Example: pip install carla==0.9.15"
    echo ""
    exit 1
else
    echo "[OK] 'carla' library is installed."
fi
echo ""

echo "[3/3] Running Live CARLA Optuna Optimizer..."
echo "Please ensure that the CARLA Simulator is running in the background."
echo ""
python3 carla_optuna_optimizer.py

echo ""
echo "==================================================="
echo "Optimization Finished! Check the 'results/' folder."
echo "==================================================="
