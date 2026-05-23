'''
SafeBench + YOLO3D Perception-Control Closed-Loop Integration Runner
This script initializes SafeBench using our YOLO3D Agent, executes dynamic scenarios,
calculates the Perception-Control Gap Score, and evaluates safety-critical edge cases.
'''

import os
import sys
import numpy as np
import cv2
import time
import argparse
import importlib.util

# Register SafeBench path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAFEBENCH_ROOT = os.path.join(PROJECT_ROOT, "SafeBench")
if SAFEBENCH_ROOT not in sys.path:
    sys.path.insert(0, SAFEBENCH_ROOT)

from evaluator import YoloEvaluator
from risk_calculator import RiskCalculator

# SafeBench imports (wrapped in try-except to allow --demo execution without carla python package)
try:
    import carla
    from safebench.carla_runner import CarlaRunner
    from safebench.util.run_util import load_config
    from safebench.util.logger import Logger
    CARLA_AVAILABLE = True
except ImportError as e:
    CARLA_AVAILABLE = False
    # Define placeholder classes for demo mode
    class CarlaRunner:
        def __init__(self, agent_config, scenario_config):
            pass
        def run(self):
            pass
    class Logger:
        def __init__(self, exp_name=None, output_dir=None):
            pass
        def log(self, msg, color=None):
            print(f"[LOG] {msg}")
    def load_config(path):
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    print(f"[WARNING] SafeBench imports failed. Running in fallback mode. Error: {e}")


class Yolo3dCarlaRunner(CarlaRunner):
    """
    Custom CarlaRunner for SafeBench that intercepts the simulation loop
    to calculate real-time Physical Risk, Perceived Risk, and cumulative Perception-Control Gap.
    """
    def __init__(self, agent_config, scenario_config):
        if not CARLA_AVAILABLE:
            raise RuntimeError("Live CARLA mode requested but CARLA Python API or SafeBench dependencies are not installed.")
        super().__init__(agent_config, scenario_config)
        self.risk_calculator = RiskCalculator()
        self.gap_records = []
        self.final_edge_case_scores = {}

    def eval(self, data_loader):
        import copy
        num_finished_scenario = 0
        data_loader.reset_idx_counter()
        
        while len(data_loader) > 0:
            sampled_scenario_configs, num_sampled_scenario = data_loader.sampler()
            num_finished_scenario += num_sampled_scenario

            # Initialize SafeBench env with configs
            static_obs = self.env.get_static_obs(sampled_scenario_configs)
            self.scenario_policy.load_model(sampled_scenario_configs)
            scenario_init_action, _ = self.scenario_policy.get_init_action(static_obs, deterministic=True)
            obs, infos = self.env.reset(sampled_scenario_configs, scenario_init_action)

            # Register ego vehicle
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos)

            # Track metrics per scenario
            score_list = {s_i: [] for s_i in range(num_sampled_scenario)}
            gap_integrals = {s_i: 0.0 for s_i in range(num_sampled_scenario)}
            collision_occurred = {s_i: False for s_i in range(num_sampled_scenario)}
            
            dt = self.fixed_delta_seconds # typically 0.05s

            self.logger.log(">> Closed-loop simulation started. Monitoring YOLO3D and Risk Calculator...")
            
            while not self.env.all_scenario_done():
                # 1. Get action from YOLO3D Agent (containing AEB and detection results)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=True)
                scenario_actions = self.scenario_policy.get_action(obs, infos, deterministic=True)

                # 2. Advance CARLA simulation step
                next_obs, rewards, dones, next_infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)

                # 3. Intercept and calculate real-time risks for each active scenario
                for idx, info in enumerate(next_infos):
                    scenario_id = info['scenario_id']
                    if self.env.finished_env[scenario_id]:
                        continue

                    # Extract ground truth of ego vehicle & obstacles from CARLA simulator directly
                    env_instance = self.env.env_list[scenario_id]._env
                    ego_vehicle = env_instance.ego_vehicle
                    
                    if ego_vehicle is None:
                        continue
                    
                    ego_transform = ego_vehicle.get_transform()
                    ego_pos = [ego_transform.location.x, ego_transform.location.y, ego_transform.location.z]
                    ego_vel_raw = ego_vehicle.get_velocity()
                    ego_vel = [ego_vel_raw.x, ego_vel_raw.y, ego_vel_raw.z]

                    # Find nearby actors in CARLA world
                    world_actors = self.world.get_actors()
                    gt_obstacles = []
                    
                    # Track vehicles and pedestrians around the ego vehicle
                    for actor in world_actors:
                        if actor.id == ego_vehicle.id:
                            continue
                        
                        dist = actor.get_location().distance(ego_transform.location)
                        if dist > 50.0: # Only evaluate within 50m
                            continue
                            
                        actor_class = 'unknown'
                        mu = 1.0
                        if 'vehicle' in actor.type_id:
                            actor_class = 'car'
                            mu = 1.0
                        elif 'walker' in actor.type_id:
                            actor_class = 'pedestrian'
                            mu = 1.8
                        elif 'traffic_light' in actor.type_id:
                            actor_class = 'traffic_light'
                            mu = 2.0
                        else:
                            continue

                        actor_vel_raw = actor.get_velocity()
                        gt_obstacles.append({
                            'class': actor_class,
                            'pos': [actor.get_location().x, actor.get_location().y, actor.get_location().z],
                            'vel': [actor_vel_raw.x, actor_vel_raw.y, actor_vel_raw.z],
                            'mu': mu
                        })

                    # Calculate multi-risk using RiskCalculator
                    yolo_detections = ego_actions[idx]['od_result']
                    
                    if len(gt_obstacles) > 0:
                        r_perc, r_gt, gap, detail_info = self.risk_calculator.calculate_multi_risk(
                            ego_pos=ego_pos, ego_vel=ego_vel,
                            gt_obstacles=gt_obstacles, yolo_detections=yolo_detections
                        )
                        
                        # Accumulate the Perception Gap over time (integration)
                        gap_integrals[scenario_id] += gap * dt
                        
                        # Log critical safety-illusion frame
                        if gap > 1.5:
                            self.logger.log(
                                f"   [WARNING] Safety Illusion Frame in Scenario {scenario_id}: "
                                f"GT-Risk={r_gt:.2f}, Perceived-Risk={r_perc:.2f}, Gap={gap:.2f}"
                            )

                    # Monitor collision sensor
                    if len(env_instance.collision_hist) > 0:
                        collision_occurred[scenario_id] = True

                    # Accumulate planning rewards
                    score = rewards[idx] if self.scenario_category == 'planning' else 1.0
                    score_list[scenario_id].append(score)

                obs = copy.deepcopy(next_obs)
                infos = copy.deepcopy(next_infos)

            # Clean up scenarios
            self.env.clean_up()

            # Calculate and print final edge case metrics for this batch
            self.logger.log("\n========================================================")
            self.logger.log("  YOLO3D + SafeBench Evaluation Batch Results")
            self.logger.log("========================================================")
            
            for s_id in score_list.keys():
                mean_reward = np.mean(score_list[s_id])
                total_gap_integral = gap_integrals[s_id]
                c_flag = collision_occurred[s_id]
                
                # J = (1.0 - Safety_Score) * Gap_Integral
                # Safety Score is 1.0 if no collision, 0.0 if collision occurred
                safety_score = 0.0 if c_flag else 1.0
                j_score = (1.0 - safety_score) * total_gap_integral

                self.logger.log(f"Scenario ID: {s_id}")
                self.logger.log(f"  - Collision Occurred: {c_flag}")
                self.logger.log(f"  - Route Mean Reward: {mean_reward:.4f}")
                self.logger.log(f"  - Cumulative Perception Gap (integral dt): {total_gap_integral:.4f}")
                self.logger.log(f"  - Critical Edge Case Score (J = Safety_Fail * Gap): {j_score:.4f}", 'red' if j_score > 0 else 'green')
                
                self.final_edge_case_scores[s_id] = {
                    'collision': c_flag,
                    'mean_reward': mean_reward,
                    'gap_integral': total_gap_integral,
                    'edge_case_score': j_score
                }
            self.logger.log("========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Run SafeBench with YOLO3D Agent Integration")
    parser.add_argument("--map", type=str, default="Town01", help="CARLA Map to load (default: Town01)")
    parser.add_argument("--scenario_category", type=str, default="planning", choices=["planning", "perception"], help="Scenario category")
    parser.add_argument("--port", type=int, default=2000, help="CARLA simulator port")
    parser.add_argument("--demo", action="store_true", help="Run in mock/demo mode if CARLA server is not available")
    args = parser.parse_args()

    # Load SafeBench configurations
    agent_cfg_path = os.path.join(SAFEBENCH_ROOT, "safebench/agent/config/yolo3d.yaml")
    
    if not os.path.exists(agent_cfg_path):
        print(f"Error: Agent config not found at {agent_cfg_path}")
        return

    agent_config = load_config(agent_cfg_path)
    
    # Standard dummy scenario config for execution
    scenario_config = {
        'seed': 42,
        'exp_name': 'safebench_yolo3d_eval',
        'output_dir': 'results/safebench_eval',
        'mode': 'eval',
        'save_video': False,
        'render': True,
        'num_scenario': 1,
        'fixed_delta_seconds': 0.05,
        'scenario_category': args.scenario_category,
        'continue_agent_training': False,
        'continue_scenario_training': False,
        'port': args.port,
        'auto_ego': False,
        'ROOT_DIR': PROJECT_ROOT,
        'tm_port': 8000,
        'policy_type': 'ordinary' if args.scenario_category == 'planning' else 'yolo',
        'max_episode_step': 200,
    }

    if args.demo or not CARLA_AVAILABLE:
        print("\n========================================================")
        print("  DEMO MODE: Simulating YOLO3D + SafeBench Loop")
        print("========================================================")
        print("Evaluating YOLO3D Agent in a mock environment loop...")
        
        # Simulating dummy camera frame
        dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Draw a dummy pedestrian at 12m
        cv2.putText(dummy_img, "Pedestrian Mock Frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Dynamically import Yolo3dAgent without executing packages that require carla nav agents
        yolo3d_agent_path = os.path.join(SAFEBENCH_ROOT, "safebench/agent/yolo3d_agent.py")
        spec = importlib.util.spec_from_file_location("safebench.agent.yolo3d_agent", yolo3d_agent_path)
        yolo3d_agent_mod = importlib.util.module_from_spec(spec)
        sys.modules["safebench.agent.yolo3d_agent"] = yolo3d_agent_mod
        spec.loader.exec_module(yolo3d_agent_mod)
        Yolo3dAgent = yolo3d_agent_mod.Yolo3dAgent
        
        class MockLogger:
            def log(self, msg, color=None):
                print(f"[MOCK LOG] {msg}")
        
        agent = Yolo3dAgent({'ego_action_dim': 2, 'model_path': ''}, MockLogger())
        
        obs = [{'img': dummy_img, 'states': np.array([0, 0, 15.0, 0])}]
        actions = agent.get_action(obs, None)
        
        print("\nDemo Output Action Control:")
        print(f"  Action: {actions[0]['ego_action']}")
        print(f"  Estimated Closest Z: {actions[0]['min_z_distance']:.2f}m")
        print("========================================================\n")
        return

    # Normal execution connecting to live CARLA server
    try:
        runner = Yolo3dCarlaRunner(agent_config, scenario_config)
        runner.run()
    except Exception as e:
        print(f"\n[ERROR] Failed to run SafeBench integration: {e}")
        print("Please verify that CARLA server is running at port 2000.")
        print("You can run this script with '--demo' to test the integration logic without CARLA.")


if __name__ == "__main__":
    main()
