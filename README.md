# W-ADAPT: Weather-Adaptive Dynamic Sensor Fusion and High-Speed Edge-Case Discovery for Autonomous Driving

> 📌 **Project Scope:** This repository establishes a robust, freeze-free pipeline to automatically sample weather environments, aggregate synchronized multi-sensor datasets, and execute post-process phases for critical edge-case discovery.

---

## Project Overview and Objectives
Conventional autonomous driving systems suffer from a significant drop in single-sensor recognition accuracy due to environmental noise such as heavy rain, fog, and sunlight. This project resolves and quantifies this issue through the following 4-step approach:

- **Scenario Definition (Phase 1)**: Defining five representative hazard scenarios commonly emphasized in autonomous driving system evaluations within the simulator.
- **Data Collection (Phase 2)**: Sampling diverse weather parameters in the CARLA simulator environment to automatically collect driving logs across hazard scenarios.
- **Weather-Adaptive Dynamic Weighting (Phase 3)**: Calculating weights for a weighted linear combination (Late Fusion) that maximizes the accuracy rate for each weather condition using mathematical optimization (SLSQP). This is based on the measurement distance error (in meters) of each sensor.
- **Critical Edge-Case Discovery (Phase 4)**: Utilizing the optimized distances to calculate a "Dynamic Risk Score Function" to automatically identify and visualize the worst-case weather domains (edge cases) that maximize the overall system hazard level.

---

## Sensor Configuration and Roles (Active Synchronization)
Although the primary perception layer relies on the **RGB Stereo Camera** with YOLO3D, this system explicitly synchronizes and logs data from the following active background sensor suite to enable Ground Truth validation and Late Fusion:
- **RGB Stereo Camera (`sensor.camera.rgb`)**: Laterally offset configuration ($y = \pm0.27\text{m}$). Resolution 1280x720, FOV 110 degrees. Primary sensor for main recognition (YOLO3D).
- **Predictive AI Model (`dist_ai`)**: A pre-trained AI model to complement distance estimation under adverse weather conditions.
- **LiDAR (`sensor.lidar.ray_cast`)**: 32 channels, 56000 pts/s. Calculates distance (`dist_lidar`) from the median value within bounding boxes.
- **Radar (`sensor.other.radar`)**: Measures relative approach velocity (`v_approach`).
- **Depth & Segmentation Cameras**: Used for acquiring Absolute Ground Truth (`dist_gt`) and object classes (`scenario_class`) processed as synchronized background queues.

---

## ⚡ Robustness & Safety Patches Implemented
To complete over 500 consecutive automated trials via Optuna without simulator deadlocks or file corruptions, the following explicit engineering solutions have been applied to the scripts:
1. **Strict Initialization Order (AttributeError Prevention)**: Guaranteed that `self.traffic_manager` is properly instantiated via the client *before* calling `set_synchronous_mode(True)`.
2. **Deadlock Auto-Escape Loop**: If the ego vehicle's velocity remains below 0.5 km/h for 5 consecutive frames after tick 15 (indicating a full stop or a severe crash), the execution loop automatically `break`s to cleanly transition to the next trial.
3. **Actor Clean Sweep (`_destroy_actors`)**: Every trial run invokes a rigorous destruction sequence for the ego-vehicle, active sensors, and injected hazard actors, preventing "ghost actors" from inducing spawn collisions in subsequent runs.
4. **Memory Contiguity Correction for Video Recording**: Fixed a silent bug where slicing the BGRA raw buffers (`[:, :, :3]`) into RGB resulted in non-contiguous memory layouts, forcing `cv2.VideoWriter` to skip frames and drop 0-byte files. The code now forces alignment using `np.ascontiguousarray()` immediately prior to writing.

---

## Verification Hazard Scenarios
Following a 40-frame (2 seconds) acceleration and cruising phase, the following hazard actors are dynamically and synchronously injected (Dynamic Hazard Injection) at specified clearance distances directly ahead.
- **Scenario A (CPNA)**: Pedestrian crossing (crosses the ego vehicle's path at 4.5 m/s).
- **Scenario B (CCRb)**: Leading vehicle sudden braking (full braking applied at the 30th frame).
- **Scenario C (CCFtap)**: Opposing vehicle suddenly cutting in for a right turn (cuts into the ego lane at the 50th frame).
- **Scenario D (AVOID)**: Static obstacle (a stationary vehicle with locked brakes is placed ahead).
- **Scenario E (RLI)**: Red light running at an intersection (a Mustang rushes in from the side with full throttle upon intersection entry).

---

## Directory Structure
```text
.
├── archive/                       # [Read-Only] Backup for old/temporary scripts
└── W-ADAPT/                       # Main directory for latest experiments and analysis
    ├── run_camera_only_experiment.py # [Active] Script for experimental loop & multi-sensor synchronization
    ├── carla_optuna_optimizer.py     # [Active] Script for automated weather sampling execution using Optuna
    ├── risk_calculator.py            # Risk function calculation module
    ├── evaluator.py                  # Evaluator for YOLO3D and AI distance predictions
    ├── phase4_edge_case_discovery.py # Core for transparent risk calculation & worst-case weather identification
    ├── plot_results.py               # Auto-plotter for 3D scatter & 2D heatmaps (for paper)
    ├── optimized_weather_weights.csv # Optimized weather-specific weight map output from Phase 3
    └── logs/                         # Storage for driving logs (trial_*.csv)
```
