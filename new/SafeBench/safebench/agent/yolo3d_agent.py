''' 
Date: 2026-05-23
Description:
    YOLOv8 3D Object Detection Agent integrating with SafeBench.
    This agent estimates obstacle distances and executes AEB (Autonomous Emergency Braking).
'''

import os
import sys
import numpy as np
import cv2

# Add project root to path to import evaluator and risk_calculator
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from evaluator import YoloEvaluator
from risk_calculator import RiskCalculator

try:
    from safebench.agent.base_policy import BasePolicy
except Exception:
    # Dynamic fallback when executing without CARLA agents dependencies
    try:
        agent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        if agent_dir not in sys.path:
            sys.path.append(agent_dir)
        from base_policy import BasePolicy
    except Exception:
        class BasePolicy:
            name = 'base'
            type = 'unlearnable'
            def __init__(self, config, logger):
                pass



class Yolo3dAgent(BasePolicy):
    name = 'yolo3d'
    type = 'unlearnable'

    def __init__(self, config, logger):
        self.logger = logger
        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.mode = 'eval'
        
        # Load YOLOv8 and Risk Calculator
        self.evaluator = YoloEvaluator()
        self.risk_calculator = RiskCalculator()
        self.logger.log(">> YOLO3D Agent initialized successfully.")

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        """
        obs: List of observation dicts from environments [{'img': ..., 'states': ...}]
        infos: List of metadata dicts
        """
        n_envs = len(obs)
        actions = []

        for i in range(n_envs):
            image = obs[i]['img']
            
            # Convert RGB image (from SafeBench) to BGR (expected by YoloEvaluator)
            bgr_image = cv2.cvtColor(np.array(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            
            # YOLO detection and 3D distance estimation
            detections, annotated_image = self.evaluator.evaluate_multi(bgr_image, return_image=True)
            
            # Convert annotated image back to RGB for visualization in SafeBench
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Determine the critical action based on obstacle classes and states
            apply_brake = False
            brake_reason = ""
            min_z = float('inf')
            
            for det in detections:
                cls = det['class']
                z = det['z_distance']
                
                if z < min_z:
                    min_z = z
                
                # AEB for physical obstacles (pedestrians, cars, barricades) within 15.0 meters
                if cls in ['pedestrian', 'car', 'construction_signal']:
                    if z <= 15.0:
                        apply_brake = True
                        brake_reason = f"Obstacle '{cls}' at {z:.2f}m"
                        break
                # Stop control for traffic lights (if RED, YELLOW, or UNKNOWN and within 18.0 meters)
                elif cls == 'traffic_light':
                    color = det.get('traffic_light_color', 'unknown')
                    if color in ['red', 'yellow', 'unknown'] and z <= 18.0:
                        apply_brake = True
                        brake_reason = f"Traffic Light [{color}] at {z:.2f}m"
                        break
            
            # ego_action control: [acceleration/brake, steer]
            if apply_brake:
                self.logger.log(f"--> [YOLO3D AEB] BRAKE applied: {brake_reason}.")
                action = np.array([-1.0, 0.0]) # Full brake
            else:
                action = np.array([0.3, 0.0])  # Normal forward throttle (0.3)
                
            actions.append({
                'ego_action': action,
                'od_result': detections,
                'annotated_image': annotated_image_rgb,
                'min_z_distance': min_z
            })

        return actions

    def load_model(self):
        pass

    def save_model(self, episode):
        pass
