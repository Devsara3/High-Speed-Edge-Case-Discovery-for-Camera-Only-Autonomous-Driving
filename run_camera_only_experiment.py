import os
import sys
import time
import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# リスク計算クラスとYOLO評価器の読み込み
from risk_calculator import RiskCalculator
from evaluator import YoloEvaluator
from carla_mock import MockCarlaEnv

from hsc_emulator import HSCEmulator

# CARLAのインポート試行
try:
    import carla
    import queue
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

class CameraOnlyExperiment:
    """
    カメラ単一センサー（Camera-Only）による標準ADASシナリオ実験走行管理クラス。
    実機CARLAおよびオフラインモック（--demo）の両方をサポートします。
    """
    def __init__(self, demo_mode=True, host='localhost', port=2000):
        self.demo_mode = demo_mode or not CARLA_AVAILABLE
        self.evaluator = YoloEvaluator()
        self.risk_calculator = RiskCalculator()
        self.hsc_emulator = HSCEmulator()
        self.camera_type = 'RGB' # デフォルトはRGB
        self.actors = []
        self.training_data = []
        self.log_data = []
        
        # HUD動画保存用
        os.makedirs('results', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter('results/sensor_record.mp4', fourcc, 20.0, (1280, 720))
        self.current_scenario = None
        self.clear_flag = False
        
        self.scenario_min_distance = float('inf')
        self.scenario_collisions = 0
        
        # シナリオ固有の進行管理変数
        self.scenario_start_x = 0.0
        self.scenario_start_loc = None  # CARLA用の開始時 Location
        self.scenario_ticks = 0
        self.trigger_dist = 0.0         # 開始位置からのトリガー距離
        
        # PIDコントローラ用パラメータ（横方向）
        self.kp_lateral = 1.0
        self.kd_lateral = 0.2
        self.ki_lateral = 0.02
        self.integral_error_lat = 0.0
        self.prev_error_lat = 0.0
        
        # PIDコントローラ用パラメータ（縦方向）
        self.kp_long = 0.8
        self.kd_long = 0.15
        self.prev_error_long = 0.0
        
        if self.demo_mode:
            print("[INFO] Running in MOCK DEMO mode (No CARLA server required).")
            base_img = "base_image.png"
            if not os.path.exists(base_img):
                # ダミー画像を作成
                dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(dummy_img, "CARLA CAMERA ONLY DEMO", (200, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.imwrite(base_img, dummy_img)
            self.mock_env = MockCarlaEnv(base_img)
        else:
            print(f"[INFO] Connecting to CARLA at {host}:{port}...")
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()
            self.actors = []
            self.ego_vehicle = None
            self.camera = None
            self.target_actor = None
            self.image_queue = queue.Queue()
            
            # 同期モード設定
            self.original_settings = self.world.get_settings()
            self.settings = self.world.get_settings()
            self.settings.synchronous_mode = True
            self.settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(self.settings)

    def _setup_mock_scenario(self, name, ego_speed_kph, gap=12.0, deceleration=-6.0):
        """
        モック環境における各シナリオの物理骨格（初期位置・速度・軌道）を初期化します。
        """
        self.mock_env.reset()
        self.time_step = 0
        self.scenario_ticks = 0
        self.scenario_start_x = 0.0
        self.clear_flag = False
        self.current_scenario = name
        self._collision_registered_this_scenario = False
        self.max_gap_this_run = -float('inf')
        self.worst_case_image = None
        self.worst_case_step = 0
        
        # Ego初期位置・速度設定
        v_ego = (ego_speed_kph / 3.6)
        self.mock_env.ego_pos = [0.0, 0.0, 0.0]
        self.mock_env.ego_vel = [v_ego, 0.0, 0.0]
        
        # ターゲットアクターのオーバーライド
        if name == 'A':
            # シナリオA (CPNA: 歩行者横断)
            v_ped = 1.39 # 5 km/h
            dist_cross = 30.0 # 横断位置 X=30m
            t_walk = 3.0 / v_ped
            d_trigger = v_ego * t_walk
            self.trigger_dist = dist_cross - d_trigger
            
            self.mock_env.obstacles = [{
                'class': 'pedestrian',
                'pos': [dist_cross, 3.0, 0.0],
                'vel': [0.0, -v_ped, 0.0],
                'mu': 1.8,
                'active': False
            }]
            print(f"[Scenario A Setup] Ego Speed: {ego_speed_kph} km/h ({v_ego:.2f} m/s). Pedestrian at X={dist_cross}m, starts walking when Ego travels {self.trigger_dist:.2f}m")
            
        elif name == 'B':
            # シナリオB (CCRb: 先行車急制動)
            self.lead_decel_started = False
            self.lead_decel_ticks = 40 # 2秒後(40ticks)に急ブレーキ
            self.lead_deceleration = deceleration
            
            self.mock_env.obstacles = [{
                'class': 'car',
                'pos': [gap, 0.0, 0.0],
                'vel': [v_ego, 0.0, 0.0],
                'mu': 1.0
            }]
            print(f"[Scenario B Setup] Ego & Lead Speed: {ego_speed_kph} km/h. Gap: {gap}m. Braking after 2.0s with {deceleration} m/s²")
            
        elif name == 'C':
            # シナリオC (CCFtap: 交差車両)
            v_target = 11.11 # 40 km/h
            x_intersect = 20.0
            t_to_intersect = x_intersect / v_ego
            x_target_start = x_intersect + v_target * t_to_intersect
            
            self.mock_env.obstacles = [{
                'class': 'car',
                'pos': [x_target_start, 3.5, 0.0], # 対向車線 Y=3.5
                'vel': [-v_target, 0.0, 0.0],
                'mu': 1.0
            }]
            print(f"[Scenario C Setup] Ego Speed: {ego_speed_kph} km/h. Oncoming starts at X={x_target_start:.2f}m with speed {v_target*3.6:.1f} km/h")
            
        elif name == 'D':
            # シナリオD (AVOID: 静的障害物回避)
            self.mock_env.obstacles = [{
                'class': 'construction_signal',
                'pos': [35.0, 0.0, 0.0], # 車線中央 X=35m
                'vel': [0.0, 0.0, 0.0],
                'mu': 1.5
            }]
            print(f"[Scenario D Setup] Ego Speed: {ego_speed_kph} km/h. Construction barrier at X=35m")
            
        elif name == 'E':
            # シナリオE (RLI: 赤信号交差点)
            self.mock_env.obstacles = [{
                'class': 'traffic_light',
                'pos': [35.0, 0.0, 0.0], # X=35m
                'vel': [0.0, 0.0, 0.0],
                'color': 'red',
                'mu': 2.0
            }]
            print(f"[Scenario E Setup] Ego Speed: {ego_speed_kph} km/h. Red Traffic Light at X=35m")

    def _setup_real_scenario(self, name, ego_speed_kph, gap=12.0, deceleration=-6.0):
        """
        CARLA実環境におけるシナリオ初期化。
        オープンワールドでAutopilotを有効化し、常に同じスポーン地点からスタートします。
        """
        self.time_step = 0
        self.scenario_ticks = 0
        self.current_scenario = name
        self.clear_flag = False
        self._collision_registered_this_scenario = False
        self.max_gap_this_run = -float('inf')
        self.worst_case_image = None
        
        spawn_points = self.world.get_map().get_spawn_points()
        # 常に同じ背景（スポーン地点）を担保
        ego_transform = spawn_points[0]
        
        ego_bp = self.blueprint_library.filter('model3')[0]
        ego_transform.location.z += 0.5
        self.ego_vehicle = self.world.spawn_actor(ego_bp, ego_transform)
        self.actors.append(self.ego_vehicle)
        self.ego_vehicle.set_autopilot(True)
        
        self.scenario_start_loc = ego_transform.location
        self.scenario_start_x = ego_transform.location.x
        
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
        self.actors.append(self.camera)
        
        self.camera.listen(lambda image: self.image_queue.put(image))
        self.world.tick()

    def run_step(self, scenario_name, target_speed_kph):
        """
        1ステップ進行し、YOLO等での距離計測とログ記録のみ行う。
        運転は完全にAutopilotに任せるため制御指令は送らない。
        """
        log_entry = {'tick': self.time_step, 'scenario': scenario_name, 'weather': 'default'}
        
        if self.demo_mode:
            log_entry['ego_vx'] = self.mock_env.ego_vel[0]
            log_entry['ego_x'] = self.mock_env.ego_pos[0]
            # モック環境の実行ロジック等 (省略)
            self.time_step += 1
            return log_entry
        else:
            self.world.tick()
            
            try:
                image = self.image_queue.get(timeout=2.0)
            except queue.Empty:
                return log_entry
                
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            frame = array[:, :, ::-1]
            
            # Autopilot駆動中の速度取得
            ego_vel = self.ego_vehicle.get_velocity()
            ego_speed = np.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
            log_entry['ego_vx'] = ego_speed
            log_entry['ego_x'] = self.ego_vehicle.get_transform().location.x
            
            # YOLO推論と距離計測
            eval_res = self.evaluator.evaluate_multi(frame, ego_speed, disparity_map=None, return_image=False)
            
            log_entry['worst_obstacle'] = 'none'
            log_entry['perception_gap'] = 0.0
            
            if len(eval_res['detections']) > 0:
                worst = eval_res['detections'][0]
                log_entry['worst_obstacle'] = worst['class']
                # 単純にAI距離やステレオ距離などを記録
                log_entry['dist_ai'] = worst.get('dist_ai', float('inf'))
                log_entry['dist_stereo'] = worst.get('dist_stereo', float('inf'))
                log_entry['dist_pinhole'] = worst.get('dist_pinhole', float('inf'))
                # GroundTruthの距離計算 (カメラから対象物への実際の距離)
                if self.target_actor:
                    t_loc = self.target_actor.get_location()
                    e_loc = self.ego_vehicle.get_location()
                    log_entry['dist_gt'] = e_loc.distance(t_loc)
                else:
                    log_entry['dist_gt'] = float('inf')
                    
                # 誤差の記録
                if log_entry['dist_gt'] != float('inf') and log_entry['dist_ai'] != float('inf'):
                    log_entry['perception_gap'] = abs(log_entry['dist_gt'] - log_entry['dist_ai'])
            
            self.log_data.append(log_entry)
            self.time_step += 1
            self.scenario_ticks += 1
            return log_entry

    def run_sequence(self):
        """
        全シナリオ(A -> B -> C -> D)を連続走行。
        """
        print("
===== Starting Open-World Dynamic Hazard Sequence =====")
        
        target_speed_kph = 40.0
        if self.demo_mode:
            self._setup_mock_scenario('A', target_speed_kph)
            self.current_scenario = 'A'
            for _ in range(100):
                self.run_step('A', target_speed_kph)
        else:
            self._setup_real_scenario('sequence', target_speed_kph)
            scenarios = ['A', 'B', 'C', 'D']
            
            print("[INFO] Initializing Autopilot... Waiting 100 ticks for good visuals.")
            for _ in range(100):
                self.run_step('sequence', target_speed_kph)
                
            for seq in scenarios:
                self.current_scenario = seq
                print(f"[INFO] Accelerating for Scenario {seq}...")
                for _ in range(40):
                    self.run_step('sequence', target_speed_kph)
                    
                self._inject_dynamic_hazard(seq)
                
                for tick in range(100):
                    # 同期モードにおける遅延アクションの適用（tickベース）
                    try:
                        import carla
                        if not self.demo_mode and hasattr(self, 'target_actor') and self.target_actor:
                            # シナリオB: 30コマ目(1.5秒後)に急ブレーキ
                            if seq == 'B' and tick == 30:
                                control = carla.VehicleControl()
                                control.brake = 1.0
                                control.throttle = 0.0
                                self.target_actor.apply_control(control)
                                print(f"[Tick {tick}] シナリオB: 先行車が急ブレーキを踏みました！")
                            # シナリオC: 20コマ目(1.0秒後)に対向車線からこちらへ急ハンドル
                            elif seq == 'C' and tick == 20:
                                control = carla.VehicleControl()
                                control.throttle = 0.6
                                control.steer = -0.5
                                self.target_actor.apply_control(control)
                                print(f"[Tick {tick}] シナリオC: 対向車が急に車線変更してきました！")
                    except ImportError:
                        pass
                        
                    log = self.run_step(seq, target_speed_kph)
                    if tick % 10 == 0:
                        print(f"DynSeq [{seq}] Step {tick}: Gap={log.get('perception_gap', 0):.2f}")
                        
                if hasattr(self, 'target_actor') and self.target_actor is not None:
                    print(f"[INFO] Clearing hazard {seq} to resume driving.")
                    self.target_actor.destroy()
                    if self.target_actor in self.actors:
                        self.actors.remove(self.target_actor)
                    self.target_actor = None
                    
                self._save_worst_image(seq)
                
        if not self.demo_mode:
            self._destroy_actors()

    def shutdown(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
            print("[INFO] Video saved to results/sensor_record.mp4")
            
        if not self.demo_mode:
            self._destroy_actors()
            self.world.apply_settings(self.original_settings)
            print("[INFO] CARLA synchronous mode disabled.")

    def export_training_data(self, filepath="results/distance_training_data.csv"):
        if not self.training_data:
            print("[WARNING] No training data collected to export.")
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame(self.training_data)
        df.to_csv(filepath, index=False)
        print(f"[SUCCESS] Exported {len(df)} distance training samples to {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera-Only ADAS Scenario Experiment Runner")
    parser.add_argument('--scenario', choices=['A', 'B', 'C', 'D', 'E', 'sequence'], default='sequence',
                        help="Scenario skeleton: A (CPNA), B (CCRb), C (CCFtap), D (AVOID), E (RLI), sequence (all dynamically)")
    parser.add_argument('--demo', action='store_true', help="Run offline in mock geometry/image mode")
    parser.add_argument('--ego-speed', type=float, default=40.0, help="Ego vehicle initial speed in km/h")
    parser.add_argument('--gap', type=float, default=12.0, help="Scenario B: Initial follow gap in meters")
    parser.add_argument('--decel', type=float, default=-6.0, help="Scenario B: Lead car deceleration in m/s2")
    parser.add_argument('--sun-alt', type=float, default=90.0, help="Sun altitude angle (0=dark, 90=noon)")
    parser.add_argument('--precip', type=float, default=0.0, help="Precipitation amount (0-100)")
    parser.add_argument('--fog', type=float, default=0.0, help="Fog density (0-100)")
    parser.add_argument('--save-path', type=str, default='results/risk_params_timeseries.png', help="Output plot path")
    args = parser.parse_args()
    
    experiment = CameraOnlyExperiment(demo_mode=args.demo)
    try:
        experiment.set_weather_params(args.sun_alt, args.precip, args.fog)
        
        if args.scenario == 'sequence':
            experiment.run_sequence()
        else:
            experiment.run_experiment(args.scenario, target_speed_kph=args.ego_speed, gap=args.gap, deceleration=args.decel)
            
        experiment.visualize_and_save(args.save_path, scenario_name=args.scenario)
        
    finally:
        experiment.shutdown()
