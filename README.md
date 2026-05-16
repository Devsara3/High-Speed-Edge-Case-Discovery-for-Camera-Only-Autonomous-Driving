# CARLA Edge Case Search & Active Learning Pipeline

This project is a pipeline designed to automatically identify "edge cases" (adverse weather conditions) where the recognition accuracy of autonomous driving AI (such as YOLOv8) declines, using an efficient search algorithm (Optuna). 

In the future, this pipeline will be extended into an **Active Learning Loop**, where the discovered edge cases are automatically used to retrain and improve the AI model, creating a self-improving autonomous driving perception system.

Currently, functionality can be verified using an image-processing-based mock environment (`carla_mock.py`) instead of the CARLA simulator.

## Pipeline Architecture

```mermaid
flowchart LR
    classDef optuna fill:#fff2cc,stroke:#ffd966,stroke-width:2px,color:#000000;
    classDef params fill:#fff2cc,stroke:#ffd966,stroke-width:2px,color:#000000;
    classDef env fill:#d9ead3,stroke:#93c47d,stroke-width:2px,color:#000000;
    classDef sensor fill:#eef4fb,stroke:#a4c2f4,stroke-width:2px,color:#000000;
    classDef yolo fill:#f9d0c4,stroke:#f28b82,stroke-width:2px,color:#000000;
    classDef save fill:#fff2cc,stroke:#ffd966,stroke-width:2px,color:#000000;
    classDef learning fill:#e6b8af,stroke:#cc0000,stroke-width:2px,color:#000000,stroke-dasharray: 5 5;

    A["Optuna Optimizer<br/>(optimizer.py)<br/>- Sun Altitude<br/>- Precipitation<br/>- Fog Density"]:::optuna
    B["CARLA Environment<br/>Mock or Real<br/>(carla_mock.py)"]:::params
    C["YOLOv8 detect (ONNX)<br/>on sensor image"]:::env
    D["Precision Metrics<br/>Count Objects<br/>Total Confidence"]:::sensor
    E["Update optimum<br/>parameters"]:::yolo
    F["Save Edge Case Image<br/>(results/)"]:::save
    G["Dataset Augmentation<br/>(Add Edge Cases)"]:::learning
    H["Retrain / Fine-tune<br/>YOLOv8 Model"]:::learning

    A --> B
    B --> C
    C --> D
    D --> E
    E -.->|Optimization Loop| A
    E -.->|If Worst Edge Case| F
    F -->|Collect Data Planned| G
    G -->|Train Planned| H
    H -->|Update Weights Planned| C
```

## Project Structure

- `optimizer.py`: The main execution script. It uses Optuna to optimize weather parameters (minimize recognition rate).
- `carla_mock.py`: Applies rain or fog effects to the base image (`base_image.png`) based on specified weather conditions.
- `evaluator.py`: Uses YOLOv8 to evaluate the number of detected objects and their confidence scores in the image.
- `visualizer.py`: Visualizes (plots) the history of search results.
- `carla_real_template.py`: A code template for connecting to the actual CARLA simulator.

## Setup

### 1. Install dependency libraries

```bash
pip install -r requirements.txt
```

### 2. Verify operation (mock environment)

First, let’s run the pipeline in a mock environment.

```bash
python optimizer.py
```

After execution, the "most difficult image to recognize (edge case)" and the "easiest image to recognize" will be saved in the `results/` directory.

## How to Integrate with the CARLA Simulator

To use this in a real CARLA environment, follow the steps below.

1. **Install the CARLA Simulator**: Download the simulator from [the official CARLA website](https://carla.org/) and launch it.
2. **Implementing the connection class**: Refer to `carla_real_template.py` to create a class that retrieves images from the actual camera sensor.
3. **Replacing the environment**: In `optimizer.py`, replace the section where `MockCarlaEnv` is imported with the new class you created.

```python
# optimizer.py
# from carla_mock import MockCarlaEnv
from my_carla_env import RealCarlaEnv

# env = MockCarlaEnv("base_image.png")
env = RealCarlaEnv()
```

## Contributions and Sharing

When pushing this repository to GitHub, be sure to include `base_image.png` so that other users can immediately verify that it works. CARLA itself is not included in the repository; it is assumed that you will set it up in your own environment.
