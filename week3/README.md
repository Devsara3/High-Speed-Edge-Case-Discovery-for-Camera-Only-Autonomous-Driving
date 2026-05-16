
# CARLA Connection and Execution Guide (For Development Team Members)

 This document provides instructions for connecting and running the ` `carla_integration_demo.py` ` file and various modules contained in this directory on the CARLA simulator.
If you are responsible for setup, please follow the steps below to configure the environment and run the program.

---

## 1. Preparing the Required Environment (Prerequisites)

 Install the CARLA Python API and the libraries required for AI inference.

```bash

pip install numpy opencv-python

pip install torch torchvision

# your CARLA server version（例: 0.9.15）
pip install carla==0.9.15
```

*(※ If  you encounter an error when running ` `pip install carla` `,  please install the`.egg` or `.whl` file located in the*  ` `*PythonAPI/carla/dist/*` `  *directory within the official CARLA folder  directly.)*

---

## 2. Starting the CARLA Server

 Before running the Python script, you must ensure **that the CARLA server (the simulator itself)** is **running**.

- **For Windows**: Open the CARLA folder in Command Prompt or a similar terminal and run the script. *(Note: If your PC has low specifications and the system is running slowly* ,  *launching it with ``CarlaUE4.exe -quality-level=Low`* `  *will improve performance.)*
    
    ```bash
    CarlaUE4.exe
    ```
    

 Once the simulator window opens and the cityscape appears, the server is ready.

---

## 3. Running the integration script and connecting

 Once the server is running, open another terminal (Command Prompt) and run the main script located in this directory.

```bash
python carla_integration_demo.py
```

### What happens when the connection is successful

1. **`Connecting to CARLA Server...`**: Establishes communication with the server using the default port `2000`.
2. **Loading AI models**: Initialization of MiDaS and other components will take place (a download may occur on the first run).
3. **Vehicle Spawn**: A Tesla Model 3 appears on the map, and a camera is mounted on the roof.
4. **Autonomous driving loop starts**: The car begins driving automatically, and camera images and depth maps are displayed in real time in a separate window (OpenCV).

---

## 4. Explanation of the "Connection Process" in the Program (For Engineers)

 These are key points for others when customizing the code. Please pay attention to the following sections in `carla_integration_demo.py`.

### ① Server connection and timeout settings

```python
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
```

- By default, CARLA uses port `2000` on `localhost`.
- **Important**: `set_timeout(10.0)` is mandatory. The default timeout of 2 seconds is too short and will cause the program to crash during the initial connection or when loading heavy maps, so we have set it to a longer duration.

### ② Enabling Synchronous Mode

```python
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS
world.apply_settings(settings)
```

- When combining AI inference (which takes a few milliseconds to tens of milliseconds) with the simulator, leaving the system asynchronous causes a phenomenon where “the simulator’s time advances too much while the AI is computing, causing the car to crash into a wall.”
- To prevent this, we enable Synchronous Mode **,** which ensures that  **"the simulator’s time does not advance even by a single millisecond untilthe Python script  calls ``world.tick()` `."**
- The value `0.05` is set to perfectly match the PID control calculation interval (dt).

### ③ Cleanup upon termination (extremely important)

```python
finally:
    settings.synchronous_mode = False
    world.apply_settings(settings)
```

- If you force-quit the program (Ctrl+C) without resetting the synchronization mode to False, **the CARLA simulator’s internal time will freeze indefinitely**, requiring a restart of CARLA itself. I have included code in `the finally block` to ensure it is reset to False.
