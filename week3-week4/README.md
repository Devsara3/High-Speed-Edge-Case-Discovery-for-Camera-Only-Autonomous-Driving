# CARLA Connection and Execution Guide (For PID Controller Tuning)

This document provides instructions for connecting, running, and tuning the vehicle's PID controllers in the CARLA simulator using the scripts in this directory. All AI dependencies (such as PyTorch or CUDA) have been removed from this control loop to ensure it runs lightweight on any development PC.

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

Before running the Python scripts, you must launch the CARLA simulator itself. To improve performance and reduce GPU load, it is recommended to launch CARLA in low-quality mode.

- **For Windows**:
    ```bash
    CarlaUE4.exe -quality-level=Low
    ```

Once the simulator window opens and the map loaded, the server is ready.

---

## 3. Running the Control Loop & Tuning Parameters

Open a separate terminal, navigate to this `week3-week4` folder, and execute the integration script. You can specify the CARLA server host, port, and custom PID gains directly as command-line arguments.

### Local Connection (Default localhost:2000)
```bash
python carla_integration_demo.py
```

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

Press **`Ctrl+C`** in the terminal to stop the simulation. The script will safely clean up the spawned vehicle and save the following analysis files in this directory:

1. **`carla_pid_tuning_results.png`**: A 4-panel telemetry plot showing speed tracking, cross-track error (lateral error), steering command, and pedal control inputs.
2. **`carla_run_recording.avi`**: A video recording of the run from the vehicle's roof camera, showing overlaid real-time telemetry (Speed, CTE, Steer).

---

## 5. Offline Preliminary Simulator (Optional)

If you want to run a quick test or study the PID tuning on a simplified mathematical vehicle model without launching CARLA, run:
```bash
python pid_experiment.py
```
This runs an offline simulation of a Kinematic Bicycle Model tracking a curved path, generating:
- **`pid_tuning_results.png`**: State comparisons of Tuned, Oscillating, and Sluggish parameters.
- **`pid_simulation.gif`**: An animated visualization of the three vehicle configurations running the path.
