#!/bin/bash

echo "==================================================="
echo "CARLA Edge Case Search Pipeline - Linux/macOS"
echo "==================================================="

echo "[1/2] Installing requirements..."
pip install -r requirements.txt

echo ""
echo "[2/2] Running Optuna Optimizer..."
python optimizer.py

echo ""
echo "==================================================="
echo "Optimization Finished! Check the 'results/' folder."
echo "==================================================="
