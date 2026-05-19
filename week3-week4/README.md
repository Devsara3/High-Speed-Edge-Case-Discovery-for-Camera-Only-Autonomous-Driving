# CARLA Connection, Execution, and Multi-Sensor Guide

This document provides instructions for connecting, running, and tuning the vehicle's PID controllers in the CARLA simulator, as well as accessing raw sensor outputs (RGB camera, Semantic Segmentation camera, LiDAR, Radar, IMU, and GNSS).

All heavier deep learning libraries (like PyTorch or CUDA) have been removed, enabling the scripts in this directory to run lightweight on any development PC.

---

## 1. Prerequisites (Required Environment)

Ensure your Python environment has the necessary libraries for mathematical plotting and image processing.

```bash
pip install numpy opencv-python matplotlib

# Install the CARLA Python API corresponding to your CARLA Server version (e.g., 0.9.15)
pip install carla==0.9.15
```

*(※ If you encounter an error running `pip install carla`, please install the `.egg` or `.whl` file located in the `PythonAPI/carla/dist/` directory within your official CARLA installation folder.)*

---

## 2. Starting the CARLA Simulator Server

Before running the Python scripts, launch the CARLA simulator itself. It is recommended to launch CARLA in low-quality mode to reduce GPU overhead.

- **For Windows**:
    ```bash
    CarlaUE4.exe -quality-level=Low
    ```

---

## 3. Running the Control Loop & Multi-Sensor Dashboard

Open a separate terminal, navigate to this `week3-week4` folder, and execute the integration script. You can specify the CARLA server host, port, and custom PID gains directly as command-line arguments.

### Local Connection (Default localhost:2000)
```bash
python carla_integration_demo.py
```
Upon running, the script will:
1. Spawn the Ego vehicle (Tesla Model 3).
2. Spawn a static obstacle vehicle (Tesla Model 3) 12 meters ahead.
3. Attach and configure 6 sensors: **RGB Camera, Semantic Segmentation Camera, LiDAR, Radar, IMU, and GNSS**.
4. Open a dual OpenCV window displaying:
   - **Left**: Raw RGB camera feed + PID control telemetry (Speed, CTE, steer/pedal commands).
   - **Right**: Color-coded semantic segmentation feed + live numerical sensor readouts (LiDAR point counts, Radar detections, IMU acceleration/compass, GNSS coordinates).

### Remote Connection
If you are running the CARLA simulator on a different machine on the local network:
```bash
python carla_integration_demo.py --host 192.168.X.X --port 2000
```

### Custom PID Gains (Tuning Example)
You can test different steering and speed control profiles by passing custom gains:
```bash
# Tuned smooth parameters:
python carla_integration_demo.py --kp-lat 0.25 --kd-lat 0.35 --kp-lon 1.5 --kd-lon 0.1
```

---

## 4. Evaluation and Results

Press **`Ctrl+C`** in the terminal to stop the simulation. The script will safely destroy all spawned actors (including the ego and obstacle vehicles) and save the following analysis files in this directory:

1. **`carla_pid_tuning_results.png`**: A 4-panel telemetry plot showing speed tracking, cross-track error (lateral error), steering command, and pedal control inputs.
2. **`carla_run_recording.avi`**: A video recording of the dual-view dashboard, capturing the camera streams, segmentation, and live telemetry.

---

## 5. Offline Preliminary Simulator (Optional)

If you want to run a quick test or study the PID tuning on a simplified mathematical vehicle model (Kinematic Bicycle Model) without launching CARLA, run:
```bash
python pid_experiment.py
```
This runs an offline simulation of a vehicle tracking a curved path, generating:
- **`pid_tuning_results.png`**: State comparisons of Tuned, Oscillating, and Sluggish parameters.
- **`pid_simulation.gif`**: An animated visualization of the three vehicle configurations running the path.
