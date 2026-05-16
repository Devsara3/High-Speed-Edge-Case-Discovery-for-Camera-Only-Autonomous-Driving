# CARLA Simulator Integration Guide

This guide provides step-by-step instructions for team members to transition the pipeline from the **Mock Environment** to the **Real CARLA Simulator**.

---

## 1. Prerequisites
- **CARLA Simulator**: Install version 0.9.13 or later from the [official CARLA releases](https://github.com/carla-simulator/carla/releases).
- **Python API**: Ensure you have the `carla` Python package installed.
  ```bash
  pip install carla
  ```

---

## 2. Setting Up the Bridge (`RealCarlaEnv`)
The pipeline is designed to be environment-agnostic. To use the real simulator, you need to complete the implementation in `carla_real_template.py`.

### Step 2.1: Spawn the Ego Vehicle & Camera
In `RealCarlaEnv.__init__` or a setup method, you must:
1. Find a blueprint for a vehicle (e.g., `model3`).
2. Spawn it at a recommended transform.
3. Attach an `rgb_camera` to the vehicle.
4. Use `.listen()` on the camera to capture images into a buffer.

### Step 2.2: Implement Ground Truth Retrieval
The `RiskCalculator` requires Ground Truth (GT) data. In `carla_real_template.py`, implement a `get_ground_truth()` method:
```python
def get_ground_truth(self):
    # Retrieve actual data from the CARLA world
    ego_transform = self.vehicle.get_transform()
    ego_velocity = self.vehicle.get_velocity()
    
    # Similarly for the target vehicle (lead car)
    target_transform = self.target_vehicle.get_transform()
    target_velocity = self.target_vehicle.get_velocity()

    return {
        'ego_pos': [ego_transform.location.x, ego_transform.location.y, ego_transform.location.z],
        'ego_vel': [ego_velocity.x, ego_velocity.y, ego_velocity.z],
        'target_pos': [target_transform.location.x, target_transform.location.y, target_transform.location.z],
        'target_vel': [target_velocity.x, target_velocity.y, target_velocity.z],
        'target_class': 'car' # Or dynamically determine
    }
```

---

## 3. Switching the Pipeline to Real CARLA
Once your `RealCarlaEnv` class is ready, update `optimizer.py`:

```python
# --- optimizer.py ---

# 1. Import your real environment
from carla_real_template import RealCarlaEnv 

if __name__ == "__main__":
    # 2. Swap MockCarlaEnv with RealCarlaEnv
    # env = MockCarlaEnv("base_image.png")
    env = RealCarlaEnv(host='127.0.0.1', port=2000)
    
    # The rest of the pipeline remains the same!
    evaluator = YoloEvaluator()
    risk_calc = RiskCalculator()
    # ...
```

---

## 4. Running the Real-World Search
1. Start the CARLA Simulator first.
2. Run the pipeline:
   ```bash
   python optimizer.py
   ```
3. The simulator weather will now change automatically during each Optuna trial, and the AI will evaluate the real sensor feed.

---

## 💡 Important Considerations
- **Synchronization**: Use CARLA's **Synchronous Mode** for reliable data collection.
  ```python
  settings = self.world.get_settings()
  settings.synchronous_mode = True
  self.world.apply_settings(settings)
  ```
- **Cleanup**: Always ensure you destroy spawned actors (vehicles/sensors) when the optimization finishes to avoid ghost actors in the next run.
