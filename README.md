# W-ADAPT: Weather-Adaptive Dynamic Sensor Fusion and High-Speed Edge-Case Discovery for Autonomous Driving

An evaluation pipeline designed to ensure the safety of autonomous driving systems under adverse weather conditions (heavy rain, dense fog, intense sunlight). It dynamically arbitrates the reliability of multiple sensors and rapidly automatically discovers and identifies Critical Edge Cases inherent in the system.

## Project Overview and Objectives
Conventional autonomous driving systems suffer from a significant drop in single-sensor recognition accuracy due to environmental noise such as heavy rain, fog, and sunlight. This project resolves and quantifies this issue through the following 4-step approach:

1. **Scenario Definition (Phase 1)**: Defining five representative hazard scenarios commonly emphasized in autonomous driving system evaluations within the simulator.
2. **Data Collection (Phase 2)**: Sampling diverse weather parameters in the CARLA simulator environment to automatically collect driving logs across hazard scenarios.
3. **Weather-Adaptive Dynamic Weighting (Phase 3)**: Calculating weights for a weighted linear combination (Late Fusion) that maximizes the accuracy rate for each weather condition using mathematical optimization (SLSQP). This is based on the measurement distance error (in meters) of each sensor.
4. **Critical Edge-Case Discovery (Phase 4)**: Utilizing the optimized distances to calculate a "Dynamic Risk Score Function" to automatically identify and visualize the worst-case weather domains (edge cases) that maximize the overall system hazard level.

---

## Sensor Configuration and Roles
This system builds a robust perception layer that does not rely on a single sensor.

- **RGB Stereo Camera (sensor.camera.rgb)**: Laterally offset configuration ($$y = \pm0.27\text{m}$$). Resolution 1280x720, FOV 110 degrees. The primary sensor responsible for main recognition (YOLO3D).
- **Predictive AI Model (dist_ai)**: A pre-trained AI model to complement distance estimation under adverse weather conditions.
- **LiDAR (sensor.lidar.ray_cast)**: 32 channels, 56000 pts/s. Calculates distance (`dist_lidar`) from the median value within bounding boxes.
- **Radar (sensor.other.radar)**: Measures relative approach velocity (`v_approach`).
- **Depth & Segmentation Cameras**: Used for acquiring Absolute Ground Truth for distance and object classes (processed as background queues).

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
    ├── run_camera_only_experiment.py # Main script for experimental loop & sensor synchronization
    ├── carla_optuna_optimizer.py     # Script for automated weather sampling execution using Optuna
    ├── risk_calculator.py            # Risk function calculation module
    ├── evaluator.py                  # Evaluator for YOLO3D and AI distance predictions
    ├── phase4_edge_case_discovery.py # Core for transparent risk calculation & worst-case weather identification
    ├── plot_results.py               # Auto-plotter for 3D scatter & 2D heatmaps (for paper)
    ├── optimized_weather_weights.csv # Optimized weather-specific weight map output from Phase 3
    └── logs/                         # Storage for driving logs (trial_*.csv)
```
