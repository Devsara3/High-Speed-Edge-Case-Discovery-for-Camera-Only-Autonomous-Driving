"""
CARLA Real-World Optuna Optimizer
本物のCARLAシミュレータに接続し、同期モードで物理天気パラメータをベイズ最適化（Optuna）を用いて直接操作、
単眼3D物体検出の複数障害物（歩行者、工事標識、他車、信号の色）に対応した脆弱性パイプライン。
"""

import argparse
import carla
import optuna
import numpy as np
import cv2
import time
import os
import queue
import pandas as pd
from evaluator import YoloEvaluator
from risk_calculator import RiskCalculator

class RealCarlaEnv:
    """
    本物のライブCARLAシミュレータへの同期モード接続およびコントロールを管理する環境クラス。
    """
    def __init__(self, host='localhost', port=2000):
        print(f"Connecting to CARLA Server at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        # クリーンアップ用のアクター追跡リスト
        self.actors = []
        self.ego_vehicle = None
        self.target_vehicles = []  # 複数車両
        self.pedestrian = None
        self.construction_barrier = None
        self.camera = None
        
        # 信号機の制御用
        self.traffic_light_color = 'red'
        
        # スレッドセーフな同期キューの初期化
        self.image_queue = queue.Queue()
        
        # 元の設定をバックアップして同期モードを有効化
        self.original_settings = self.world.get_settings()
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05 # 20 FPS
        self.world.apply_settings(self.settings)
        print("Synchronous mode enabled (dt = 0.05s).")

        # シナリオアクターのセットアップ
        self._setup_scenario()

    def _setup_scenario(self):
        """
        評価用の固定シナリオ（複数障害物・信号機配置）をロードしてアクターをスポーンします。
        """
        try:
            # 1. 自車 (Ego Vehicle: Tesla Model 3) のスポーン
            ego_bp = self.blueprint_library.filter('model3')[0]
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points found on the map.")
            
            ego_transform = spawn_points[0]
            ego_transform.location.z += 0.5
            
            self.ego_vehicle = self.world.spawn_actor(ego_bp, ego_transform)
            self.actors.append(self.ego_vehicle)
            print(f"Ego vehicle (Tesla Model 3) spawned at {ego_transform.location}")

            # 2. 先行車 1 (一般車: 25m前方)
            lead_bp = self.blueprint_library.filter('model3')[0]
            fwd = ego_transform.get_forward_vector()
            lead_loc = ego_transform.location + fwd * 25.0
            lead_loc.z += 0.5
            lead_transform = carla.Transform(lead_loc, ego_transform.rotation)
            lead_car = self.world.spawn_actor(lead_bp, lead_transform)
            self.actors.append(lead_car)
            self.target_vehicles.append(lead_car)
            
            # 3. 先行車 2 (大型トラック: 35m前方)
            truck_bp = self.blueprint_library.filter('carlacola')[0]
            truck_loc = ego_transform.location + fwd * 35.0
            truck_loc.z += 0.5
            truck_transform = carla.Transform(truck_loc, ego_transform.rotation)
            truck_car = self.world.spawn_actor(truck_bp, truck_transform)
            self.actors.append(truck_car)
            self.target_vehicles.append(truck_car)

            # 4. 歩行者 (Pedestrian: 10m前方、少し左側にスポーンして横断開始)
            ped_bp = self.blueprint_library.filter('walker.pedestrian.*')[0]
            right_vector = ego_transform.get_right_vector()
            ped_loc = ego_transform.location + fwd * 10.0 - right_vector * 3.0
            ped_loc.z += 0.5
            ped_transform = carla.Transform(ped_loc, ego_transform.rotation)
            self.pedestrian = self.world.spawn_actor(ped_bp, ped_transform)
            self.actors.append(self.pedestrian)
            
            # 歩行者に移動制御を追加 (横断歩道を渡る挙動)
            ped_control = carla.WalkerControl()
            ped_control.direction = right_vector
            ped_control.speed = 1.0 # 1m/s
            self.pedestrian.apply_control(ped_control)
            print(f"Pedestrian spawned crossing the road.")

            # 5. 工事用バリケード (Construction Signal: 18m前方、右側の路肩)
            barrier_bp = self.blueprint_library.find('static.prop.constructioncone')
            barrier_loc = ego_transform.location + fwd * 18.0 + right_vector * 2.0
            barrier_loc.z += 0.2
            barrier_transform = carla.Transform(barrier_loc, ego_transform.rotation)
            self.construction_barrier = self.world.spawn_actor(barrier_bp, barrier_transform)
            self.actors.append(self.construction_barrier)

            # 6. カメラセンサーの取り付け (フロントガラス上部)
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '1280')
            camera_bp.set_attribute('image_size_y', '720')
            camera_bp.set_attribute('fov', '110')
            
            camera_transform = carla.Transform(
                carla.Location(x=2.0, y=0.0, z=1.4),
                carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0)
            )
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
            self.actors.append(self.camera)

            # カメラ画像コールバック
            def _on_camera_capture(image):
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                bgr_image = array[:, :, :3]
                self.image_queue.put(bgr_image)

            self.camera.listen(_on_camera_capture)
            
            # 旁観者視点アップデート
            self._update_spectator()

        except Exception as e:
            self.destroy()
            raise e

    def _update_spectator(self):
        if self.ego_vehicle is None:
            return
        ego_loc = self.ego_vehicle.get_location()
        spectator_location = carla.Location(
            x=ego_loc.x - 15.0,
            y=ego_loc.y,
            z=ego_loc.z + 8.0
        )
        spectator_rotation = carla.Rotation(pitch=-15.0, yaw=0.0, roll=0.0)
        self.world.get_spectator().set_transform(
            carla.Transform(spectator_location, spectator_rotation)
        )

    def set_weather(self, sun_altitude_angle, precipitation, fog_density):
        """
        CARLAの環境物理パラメータを設定。
        """
        weather = carla.WeatherParameters(
            sun_altitude_angle=sun_altitude_angle,
            precipitation=precipitation,
            fog_density=fog_density,
            cloudiness=max(precipitation, fog_density),
            wetness=precipitation,
            wind_intensity=10.0
        )
        self.world.set_weather(weather)
        self.world.tick()

    def set_traffic_light_color(self, color):
        """
        車両前方の最寄り信号機アクターを制御します。
        """
        self.traffic_light_color = color
        ego_loc = self.ego_vehicle.get_location()
        
        # 自車に一番近い信号機アクターを検索
        traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        closest_light = None
        min_dist = float('inf')
        
        for tl in traffic_lights:
            dist = tl.get_location().distance(ego_loc)
            if dist < min_dist:
                min_dist = dist
                closest_light = tl
                
        if closest_light is not None:
            state = carla.TrafficLightState.Green
            if color == 'red':
                state = carla.TrafficLightState.Red
            elif color == 'yellow':
                state = carla.TrafficLightState.Yellow
                
            closest_light.set_state(state)
            closest_light.set_red_time(100.0)  # テスト期間中に変わらないように固定
            closest_light.set_yellow_time(100.0)
            closest_light.set_green_time(100.0)
            # 設定確定のため同期更新
            self.world.tick()

    def get_image(self):
        """
        最新の画像フレームを取得。
        """
        self.world.tick()
        try:
            bgr_image = self.image_queue.get(timeout=2.0)
            while not self.image_queue.empty():
                bgr_image = self.image_queue.get_nowait()
            return bgr_image
        except queue.Empty:
            print("Warning: Camera frame queue timeout. Returning dummy image.")
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    def get_ground_truth(self):
        """
        CARLAシミュレータから直接物理真値（Ground Truth）のリストを取得。
        """
        ego_transform = self.ego_vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_vel = self.ego_vehicle.get_velocity()
        
        obstacles = []
        
        # 1. 歩行者
        if self.pedestrian is not None and self.pedestrian.is_alive:
            ped_loc = self.pedestrian.get_location()
            ped_vel = self.pedestrian.get_velocity()
            obstacles.append({
                'class': 'pedestrian',
                'pos': [ped_loc.x, ped_loc.y, ped_loc.z],
                'vel': [ped_vel.x, ped_vel.y, ped_vel.z],
                'mu': 1.8
            })
            
        # 2. 先行車両
        for idx, vehicle in enumerate(self.target_vehicles):
            if vehicle.is_alive:
                v_loc = vehicle.get_location()
                v_vel = vehicle.get_velocity()
                obstacles.append({
                    'class': 'car',
                    'pos': [v_loc.x, v_loc.y, v_loc.z],
                    'vel': [v_vel.x, v_vel.y, v_vel.z],
                    'mu': 1.0 if idx == 0 else 1.2
                })
                
        # 3. 工事用バリケード
        if self.construction_barrier is not None and self.construction_barrier.is_alive:
            b_loc = self.construction_barrier.get_location()
            obstacles.append({
                'class': 'construction_signal',
                'pos': [b_loc.x, b_loc.y, b_loc.z],
                'vel': [0.0, 0.0, 0.0],
                'mu': 1.5
            })
            
        # 4. 信号機
        # 仮想の信号機真値位置として自車の20m前方を定義
        fwd = ego_transform.get_forward_vector()
        tl_pos = ego_loc + fwd * 20.0
        tl_mu = 2.0 if self.traffic_light_color in ['red', 'yellow'] else 0.0
        obstacles.append({
            'class': 'traffic_light',
            'pos': [tl_pos.x, tl_pos.y, tl_pos.z + 5.0],
            'vel': [0.0, 0.0, 0.0],
            'color': self.traffic_light_color,
            'mu': tl_mu
        })
        
        return {
            'ego_pos': [ego_loc.x, ego_loc.y, ego_loc.z],
            'ego_vel': [ego_vel.x, ego_vel.y, ego_vel.z],
            'obstacles': obstacles
        }

    def reset_actors_physics(self):
        """
        アクターの位置と速度を初期化。
        """
        spawn_points = self.world.get_map().get_spawn_points()
        ego_transform = spawn_points[0]
        ego_transform.location.z += 0.5
        self.ego_vehicle.set_transform(ego_transform)
        fwd = ego_transform.get_forward_vector()
        self.ego_vehicle.set_target_velocity(carla.Vector3D(x=fwd.x * 15.0, y=fwd.y * 15.0, z=fwd.z * 15.0))
        
        # 他アクターも再配置
        # 先行車 1
        lead_loc = ego_transform.location + fwd * 25.0
        lead_loc.z += 0.5
        self.target_vehicles[0].set_transform(carla.Transform(lead_loc, ego_transform.rotation))
        self.target_vehicles[0].set_target_velocity(carla.Vector3D(x=fwd.x * 10.0, y=fwd.y * 10.0, z=fwd.z * 10.0))
        
        # トラック
        truck_loc = ego_transform.location + fwd * 35.0
        truck_loc.z += 0.5
        self.target_vehicles[1].set_transform(carla.Transform(truck_loc, ego_transform.rotation))
        self.target_vehicles[1].set_target_velocity(carla.Vector3D(0, 0, 0))
        
        # 歩行者
        right_vector = ego_transform.get_right_vector()
        ped_loc = ego_transform.location + fwd * 10.0 - right_vector * 3.0
        ped_loc.z += 0.5
        self.pedestrian.set_transform(carla.Transform(ped_loc, ego_transform.rotation))
        ped_control = carla.WalkerControl()
        ped_control.direction = right_vector
        ped_control.speed = 1.0
        self.pedestrian.apply_control(ped_control)
        
        # 工事用バリケード
        barrier_loc = ego_transform.location + fwd * 18.0 + right_vector * 2.0
        barrier_loc.z += 0.2
        self.construction_barrier.set_transform(carla.Transform(barrier_loc, ego_transform.rotation))

        for _ in range(5):
            self.world.tick()
        
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break
        
        self._update_spectator()

    def destroy(self):
        print("Cleaning up CARLA real actors and restoring settings...")
        for actor in self.actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actors = []
        self.target_vehicles = []
        
        try:
            self.world.apply_settings(self.original_settings)
            print("CARLA original settings restored successfully.")
        except Exception as e:
            print(f"Error restoring CARLA settings: {e}")

def run_real_carla_optimization(n_trials=30, sampler_name='TPE', traffic_light_color='red'):
    """
    ライブCARLAシミュレータでOptuna最適化ループを実行し、標準RGBカメラの脆弱性を評価します。
    """
    os.makedirs("results", exist_ok=True)
    
    env = RealCarlaEnv()
    evaluator = YoloEvaluator()
    risk_calculator = RiskCalculator()
    
    if sampler_name == 'Random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
        
    study = optuna.create_study(direction="maximize", sampler=sampler)
    history = []
    
    worst_gap = -float('inf')
    
    try:
        def objective(trial):
            nonlocal worst_gap
            start_time = time.time()
            
            env.reset_actors_physics()
            
            sun_altitude_angle = trial.suggest_float("sun_altitude_angle", -15.0, 90.0)
            precipitation = trial.suggest_float("precipitation", 0.0, 100.0)
            fog_density = trial.suggest_float("fog_density", 0.0, 100.0)
            
            # 環境設定
            env.set_weather(sun_altitude_angle, precipitation, fog_density)
            env.set_traffic_light_color(traffic_light_color)
            
            # 近接動作を安定させるため同期モードで15Tick進行
            for _ in range(15):
                env.get_image()
            
            # 最新カメラ画像の取得
            img = env.get_image()
            gt = env.get_ground_truth()
            
            # RGBの推論とリスク評価
            detections, annotated = evaluator.evaluate_multi(img, return_image=True)
            r_perc, r_gt, gap, info = risk_calculator.calculate_multi_risk(
                ego_pos=gt['ego_pos'], ego_vel=gt['ego_vel'],
                gt_obstacles=gt['obstacles'], yolo_detections=detections
            )
            
            if gap > worst_gap:
                worst_gap = gap
                cv2.imwrite(f"results/real_edge_case_worst_{sampler_name}_{traffic_light_color}.jpg", annotated)
                print(f"--> [NEW WORST CARLA EDGE CASE] Gap Score: {gap:.4f}")
                print(f"    Params: Sun={sun_altitude_angle:.2f}, Rain={precipitation:.2f}, Fog={fog_density:.2f}")
                
            elapsed_time = time.time() - start_time
            
            history.append({
                "trial": trial.number,
                "sun_altitude_angle": sun_altitude_angle,
                "precipitation": precipitation,
                "fog_density": fog_density,
                "traffic_light_color": traffic_light_color,
                "r_gt": r_gt,
                "r_perceived": r_perc,
                "gap": gap,
                "worst_obstacle": info['worst_obstacle'],
                "elapsed_time_sec": elapsed_time
            })
            
            return gap

        study.optimize(objective, n_trials=n_trials)
        
    finally:
        env.destroy()

    # CSV書き出し
    df_history = pd.DataFrame(history)
    df_history.to_csv(f"results/real_history_{sampler_name}_{traffic_light_color}.csv", index=False)
    
    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA + Optuna multi-obstacle edge-case search")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials (default: 30)")
    parser.add_argument("--sampler", type=str, default="TPE", choices=["TPE", "Random"], help="Sampler (default: TPE)")
    parser.add_argument("--color", type=str, default="red", choices=["red", "green", "yellow"], help="Traffic light color (default: red)")
    args = parser.parse_args()

    print("====================================================")
    print("  CARLA Live Optuna Multi-Obstacle Edge-Case Search")
    print(f"  Trials: {args.trials}, Sampler: {args.sampler}, Signal Color: {args.color.upper()}")
    print("====================================================")
    
    try:
        study = run_real_carla_optimization(n_trials=args.trials, sampler_name=args.sampler, traffic_light_color=args.color)
        print(f"\n{args.sampler} CARLA Optimization completed!")
        print(f"Worst Edge Case (Safety Illusion) Gap Score: {study.best_trial.value:.4f}")
        print(f"Parameters: {study.best_trial.params}")
    except Exception as e:
        print(f"\nFailed to execute live CARLA pipeline: {e}")
        print("Please verify that CARLA server is running and python-carla is installed.")
