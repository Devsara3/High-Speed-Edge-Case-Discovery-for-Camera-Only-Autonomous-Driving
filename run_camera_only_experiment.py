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
        self.log_data = []
        self.training_data = []
        self.time_step = 0
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
        実機CARLAシミュレータにおける各シナリオの物理骨格（アクター配置、速度制御初期化）
        """
        self._destroy_actors()
        
        self.time_step = 0
        self.scenario_ticks = 0
        self.clear_flag = False
        self.current_scenario = name
        self._collision_registered_this_scenario = False
        self.max_gap_this_run = -float('inf')
        self.worst_case_image = None
        self.worst_case_step = 0
        
        spawn_points = self.world.get_map().get_spawn_points()
        ego_transform = spawn_points[0]
        target_tl = None
        
        # Scenario E の場合、事前に信号機がある交差点の手前のスポーンポイントを探す
        if name == 'E':
            best_spawn = spawn_points[0]
            for sp in spawn_points:
                wp = self.world.get_map().get_waypoint(sp.location)
                found = False
                for _ in range(20): # 最大40m前方を探索
                    next_wps = wp.next(2.0)
                    if not next_wps:
                        break
                    wp = next_wps[0]
                    # 前方のwaypoint周辺(15m以内)に信号機があるかチェック
                    for tl in self.world.get_actors().filter('traffic.traffic_light'):
                        if tl.get_location().distance(wp.transform.location) < 15.0:
                            best_spawn = sp
                            target_tl = tl
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            ego_transform = best_spawn
        
        # 1. 自車のスポーン
        ego_bp = self.blueprint_library.filter('model3')[0]
        ego_transform.location.z += 0.5
        self.ego_vehicle = self.world.spawn_actor(ego_bp, ego_transform)
        self.actors.append(self.ego_vehicle)
        
        # 開始位置の記録
        self.scenario_start_loc = ego_transform.location
        self.scenario_start_x = ego_transform.location.x
        
        # 2. カメラの取り付け
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
        
        # キューのクリア
        while not self.image_queue.empty():
            self.image_queue.get()
            
        # コールバックで(フレームID, 画像)を格納
        def _on_camera_capture(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            bgr_image = array[:, :, :3]
            self.image_queue.put((image.frame, bgr_image))
            
        self.camera.listen(_on_camera_capture)
        
        v_ego = ego_speed_kph / 3.6
        fwd = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        
        if name == 'A':
            ped_bp = self.blueprint_library.filter('walker.pedestrian.*')[0]
            ped_loc = ego_transform.location + fwd * 30.0 - right * 3.0
            ped_loc.z += 0.5
            ped_transform = carla.Transform(ped_loc, ego_transform.rotation)
            self.target_actor = self.world.spawn_actor(ped_bp, ped_transform)
            self.actors.append(self.target_actor)
            
            v_ped = 1.39
            t_walk = 3.0 / v_ped
            d_trigger = v_ego * t_walk
            self.trigger_dist = 30.0 - d_trigger
            
        elif name == 'B':
            lead_bp = self.blueprint_library.filter('model3')[0]
            lead_loc = ego_transform.location + fwd * gap
            lead_loc.z += 0.5
            lead_transform = carla.Transform(lead_loc, ego_transform.rotation)
            self.target_actor = self.world.spawn_actor(lead_bp, lead_transform)
            self.actors.append(self.target_actor)
            
            self.lead_decel_started = False
            self.lead_decel_ticks = 40
            self.lead_deceleration = deceleration
            
        elif name == 'C':
            lead_bp = self.blueprint_library.filter('model3')[0]
            v_target = 11.11
            x_intersect = 20.0
            t_to_intersect = x_intersect / v_ego
            x_target_start = x_intersect + v_target * t_to_intersect
            
            target_loc = ego_transform.location + fwd * x_target_start + right * 3.5
            target_loc.z += 0.5
            target_rot = carla.Rotation(pitch=ego_transform.rotation.pitch,
                                        yaw=ego_transform.rotation.yaw + 180.0,
                                        roll=ego_transform.rotation.roll)
            target_transform = carla.Transform(target_loc, target_rot)
            self.target_actor = self.world.spawn_actor(lead_bp, target_transform)
            self.actors.append(self.target_actor)
            
        elif name == 'D':
            barrier_bp = self.blueprint_library.find('static.prop.constructioncone')
            barrier_loc = ego_transform.location + fwd * 35.0
            barrier_loc.z += 0.2
            barrier_transform = carla.Transform(barrier_loc, ego_transform.rotation)
            self.target_actor = self.world.spawn_actor(barrier_bp, barrier_transform)
            self.actors.append(self.target_actor)

        elif name == 'E':
            if target_tl:
                target_tl.set_state(carla.TrafficLightState.Red)
                target_tl.freeze(True)
                self.target_actor = target_tl
            else:
                print("[WARNING] Could not find a valid traffic light for Scenario E. Using dummy fallback.")
                barrier_bp = self.blueprint_library.find('static.prop.streetbarrier')
                barrier_loc = ego_transform.location + fwd * 35.0
                barrier_loc.z += 2.0
                barrier_transform = carla.Transform(barrier_loc, ego_transform.rotation)
                self.target_actor = self.world.spawn_actor(barrier_bp, barrier_transform)
                self.actors.append(self.target_actor)

        # 初期馴染ませ用tick
        for _ in range(5):
            self.world.tick()

        # 馴染ませ用tickでキューに溜まった古い画像をすべて取り出して破棄する
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break
        
        # 最初のステップのために、最初の world.tick() を呼んでおく
        self.next_frame_id = self.world.tick()
        
        # 旁观者视角：车尾后上方
        self._update_spectator()

    def _update_spectator(self):
        if not hasattr(self, 'ego_vehicle') or self.ego_vehicle is None:
            return
        ego_tf = self.ego_vehicle.get_transform()
        fwd = ego_tf.get_forward_vector()
        spec_loc = carla.Location(
            x=ego_tf.location.x - fwd.x * 12.0,
            y=ego_tf.location.y - fwd.y * 12.0,
            z=ego_tf.location.z + 8.0
        )
        spec_rot = carla.Rotation(pitch=-20.0, yaw=ego_tf.rotation.yaw, roll=0.0)
        self.world.get_spectator().set_transform(carla.Transform(spec_loc, spec_rot))

    def _destroy_actors(self):
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.stop()
        for actor in self.actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actors = []
        self.ego_vehicle = None
        self.camera = None
        self.target_actor = None

    def run_step(self, scenario_name, target_speed_kph):
        """
        1シミュレーションステップを実行。制御、認識（YOLO3D）、リスク計算、ロギング。
        """
        dt = 0.05
        self.time_step += 1
        self.scenario_ticks += 1
        
        # 1. 物理状態およびカメラ画像（YOLO3D）の取得
        if self.demo_mode:
            ego_pos = self.mock_env.ego_pos
            ego_vel = self.mock_env.ego_vel
            gt_data = self.mock_env.get_ground_truth()
            gt_obstacles = gt_data['obstacles']
            travel_dist = ego_pos[0] - self.scenario_start_x
            
            # シナリオ別アクター移動トリガー
            if scenario_name == 'A':
                ped_actor = self.mock_env.obstacles[0]
                if travel_dist >= self.trigger_dist:
                    ped_actor['active'] = True
                if ped_actor.get('active', False):
                    ped_actor['pos'][1] += ped_actor['vel'][1] * dt
                    if ped_actor['pos'][1] < -3.0:
                        ped_actor['pos'][1] = -3.0
                        ped_actor['vel'][1] = 0.0
            elif scenario_name == 'B':
                lead_actor = self.mock_env.obstacles[0]
                if self.scenario_ticks >= self.lead_decel_ticks:
                    self.lead_decel_started = True
                
                if self.lead_decel_started:
                    lead_vel_x = lead_actor['vel'][0] + self.lead_deceleration * dt
                    lead_actor['vel'][0] = max(0.0, lead_vel_x)
                lead_actor['pos'][0] += lead_actor['vel'][0] * dt
            elif scenario_name == 'C':
                oncoming_actor = self.mock_env.obstacles[0]
                oncoming_actor['pos'][0] += oncoming_actor['vel'][0] * dt
                
            # モック環境用の動的検出器の構築 (YOLOの天候による見落とし・距離誤差の再現)
            sun_alt = self.mock_env.sun_altitude_angle
            fog = self.mock_env.fog_density
            precip = self.mock_env.precipitation
            
            image = self.mock_env.get_image()
            _ = self.evaluator.evaluate_multi(image, ego_speed=ego_vel[0])
            
            # 幾何学座標に基づき、天候の影響を適用した認識結果を生成
            yolo_detections = []
            img_width = 1280.0
            img_height = 720.0
            fov_rad = np.radians(110.0)
            focal_length = img_width / (2.0 * np.tan(fov_rad / 2.0))
            c_y = img_height / 2.0
            
            for obs in self.mock_env.obstacles:
                obs_class = obs['class']
                obs_pos = np.array(obs['pos'], dtype=float)
                rel_pos = obs_pos - np.array(ego_pos, dtype=float)
                dist_gt = np.linalg.norm(rel_pos)
                
                if rel_pos[0] >= -2.0 and dist_gt <= 60.0:
                    visibility = 1.0 - (fog / 100.0) * 0.65 - (precip / 100.0) * 0.3
                    visibility = max(0.05, visibility)
                    brightness = max(0.1, min(1.0, (sun_alt + 15.0) / 105.0))
                    
                    det_prob = 0.98 * visibility * (0.4 + 0.6 * brightness)
                    
                    if np.random.rand() > det_prob:
                        yolo_z = float('inf')
                        yolo_rel_pos = None
                        detected_color = None
                        y_bottom = 0.0
                        height_norm = 0.0
                        width_norm = 0.0
                    else:
                        x_pred = rel_pos[1]
                        y_pred = rel_pos[2] - 1.4
                        z_pred = rel_pos[0]
                        
                        error_std = (1.0 - visibility) * 0.18 * dist_gt + 0.15
                        noise = np.random.normal(0, error_std)
                        yolo_z = z_pred + noise
                        yolo_z = max(0.1, yolo_z)
                        
                        scale = yolo_z / max(0.1, z_pred)
                        yolo_x = x_pred * scale
                        yolo_y = y_pred * scale
                        yolo_rel_pos = [yolo_x, yolo_y, yolo_z]
                        
                        detected_color = None
                        if obs_class == 'traffic_light':
                            detected_color = obs.get('color', self.mock_env.traffic_light_color)
                            if detected_color in ['red', 'yellow'] and np.random.rand() > visibility:
                                detected_color = 'green'
                                
                        # 幾何学的にBBoxサイズと位置を逆算（モック用データ収集のため）
                        real_w = 1.8
                        real_h = 1.5
                        if obs_class == 'pedestrian':
                            real_w = 0.5
                            real_h = 1.7
                        elif obs_class == 'construction_signal':
                            real_w = 0.8
                            real_h = 0.9
                            
                        w_pix = (focal_length * real_w) / yolo_z
                        h_pix = (focal_length * real_h) / yolo_z
                        
                        # 接地点 Y2 の計算 (H_cam = 1.4m)
                        pitch_rad = np.radians(-5.0)
                        phi = -np.arctan(1.4 / yolo_z)
                        ang = phi - pitch_rad
                        y2_val = c_y + focal_length * np.tan(ang)
                        
                        y_bottom = np.clip(float(y2_val) / img_height, 0.0, 1.0)
                        height_norm = np.clip(float(h_pix) / img_height, 0.0, 1.0)
                        width_norm = np.clip(float(w_pix) / img_width, 0.0, 1.0)
                                
                    yolo_detections.append({
                        'class': obs_class,
                        'confidence': 0.8 * visibility,
                        'z_distance': yolo_z,
                        'yolo3d_rel_pos': yolo_rel_pos,
                        'traffic_light_color': detected_color,
                        'bbox_y_bottom': y_bottom,
                        'bbox_height': height_norm,
                        'bbox_width': width_norm
                    })
        else:
            # 1. 期待するフレームID (self.next_frame_id) の画像をキューから取得する（同期）
            image = None
            start_wait = time.time()
            while time.time() - start_wait < 2.0:
                try:
                    sensor_frame, bgr_img = self.image_queue.get(timeout=0.1)
                    if sensor_frame == self.next_frame_id:
                        image = bgr_img
                        break
                    elif sensor_frame < self.next_frame_id:
                        # 古いフレームは破棄
                        continue
                    else:
                        # もし期待するフレームより新しいフレームが届いた場合は、
                        # 同期ズレが発生しているので、それを採用した上で next_frame_id を更新する
                        print(f"[WARNING] Sync drift detected. Expected {self.next_frame_id}, got {sensor_frame}. Adjusting sync frame.")
                        self.next_frame_id = sensor_frame
                        image = bgr_img
                        break
                except queue.Empty:
                    break
            
            if image is None:
                # タイムアウトやドロップ時は警告を出し、黒画面で補完
                print(f"[WARNING] Mismatched or dropped camera frame for tick {self.next_frame_id}. Perception will fall back.")
                image = np.zeros((720, 1280, 3), dtype=np.uint8)
                
            # 2. このフレームにおける最新の物理状態を取得する
            # これにより、取得した画像と物理状態が完全に同じフレームID (self.next_frame_id) のもので同期する！
            ego_transform = self.ego_vehicle.get_transform()
            ego_pos = [ego_transform.location.x, ego_transform.location.y, ego_transform.location.z]
            ego_vel_vec = self.ego_vehicle.get_velocity()
            ego_vel = [ego_vel_vec.x, ego_vel_vec.y, ego_vel_vec.z]
            travel_dist = ego_transform.location.distance(self.scenario_start_loc)
            
            if scenario_name == 'A':
                if travel_dist >= self.trigger_dist:
                    ped_control = carla.WalkerControl()
                    ped_control.direction = ego_transform.get_right_vector()
                    ped_control.speed = 1.39
                    self.target_actor.apply_control(ped_control)
            elif scenario_name == 'B':
                if self.scenario_ticks >= self.lead_decel_ticks:
                    self.lead_decel_started = True
                
                if self.lead_decel_started:
                    v_lead = self.target_actor.get_velocity()
                    new_vx = max(0.0, v_lead.x + self.lead_deceleration * dt)
                    self.target_actor.set_target_velocity(carla.Vector3D(new_vx, v_lead.y, v_lead.z))
                    
            elif scenario_name == 'C':
                if self.target_actor is not None and self.target_actor.is_alive:
                    ego_tf = self.ego_vehicle.get_transform()
                    target_tf = self.target_actor.get_transform()
                    dist_to_target = ego_tf.location.distance(target_tf.location)
                    
                    speed = 11.11 
                    fwd_vec = target_tf.get_forward_vector()
                    
                    # 距離が15m以下になったら、進行方向ベクトルに右方向ベクトルを足して強引に右折させる
                    if dist_to_target < 15.0:
                        right_vec = target_tf.get_right_vector()
                        # 前方と右方をブレンドして斜めの速度ベクトルにする
                        new_dir = carla.Vector3D(fwd_vec.x + right_vec.x * 0.7, 
                                                 fwd_vec.y + right_vec.y * 0.7, 
                                                 fwd_vec.z + right_vec.z * 0.7)
                        norm = (new_dir.x**2 + new_dir.y**2 + new_dir.z**2)**0.5
                        vel = carla.Vector3D(new_dir.x/norm * speed, new_dir.y/norm * speed, new_dir.z/norm * speed)
                        self.target_actor.set_target_velocity(vel)
                        # 見た目の回転も加える
                        self.target_actor.set_target_angular_velocity(carla.Vector3D(0, 0, 30.0))
                    else:
                        self.target_actor.set_target_velocity(carla.Vector3D(fwd_vec.x * speed, fwd_vec.y * speed, fwd_vec.z * speed))
            
            yolo_detections = self.evaluator.evaluate_multi(image, ego_speed=ego_vel[0])
            
            gt_obstacles = []
            if self.target_actor is not None and self.target_actor.is_alive:
                t_trans = self.target_actor.get_transform()
                t_vel_vec = self.target_actor.get_velocity()
                
                act_class = 'unknown'
                mu_val = 1.0
                if scenario_name == 'A':
                    act_class = 'pedestrian'
                    mu_val = 1.8
                elif scenario_name in ['B', 'C']:
                    act_class = 'car'
                    mu_val = 1.0
                elif scenario_name == 'D':
                    act_class = 'construction_signal'
                    mu_val = 1.5
                elif scenario_name == 'E':
                    act_class = 'traffic_light'
                    mu_val = 2.0
                    
                gt_obstacles.append({
                    'class': act_class,
                    'pos': [t_trans.location.x, t_trans.location.y, t_trans.location.z],
                    'vel': [t_vel_vec.x, t_vel_vec.y, t_vel_vec.z],
                    'mu': mu_val,
                    'color': 'red' if act_class == 'traffic_light' else None
                })

        # 2. カメラ認識（YOLO3D）のみに基づく走行制御（加減速・回避操舵）の計算
        closest_hazard = None
        min_hazard_dist = float('inf')
        offset_y = 0.0
        
        for det in yolo_detections:
            if det['yolo3d_rel_pos'] is not None:
                x_rel, y_rel, z_rel = det['yolo3d_rel_pos']
                
                # AVOID (工事用バリケード回避) 判定
                if det['class'] == 'construction_signal' and z_rel <= 18.0:
                    offset_y = 2.5
                
                # AEB ブレーキ判定: 自車走行レーン内 (|x_rel| < 1.8m) の前方障害物
                is_lane_hazard = abs(x_rel) < 1.8
                # 信号機（赤・黄）の停止判定： 進行方向(Z)にあれば止まる
                is_red_light = det['class'] == 'traffic_light' and det.get('traffic_light_color') in ['red', 'yellow']
                
                if (is_lane_hazard or is_red_light) and z_rel < min_hazard_dist:
                    min_hazard_dist = z_rel
                    closest_hazard = det
                    
        # 2-A. Longitudinal AEB 制御
        target_v = target_speed_kph / 3.6
        
        # データ収集用など、AIが未学習でフォールバックが誤作動する場合はブレーキを無視する(最低限の安全装置のみ)
        # 距離推定AI(Regressor)がロードされていない場合は、強制的にPID追従を優先
        if getattr(self.evaluator, 'distance_regressor', None) is None:
            min_hazard_dist = float('inf')  # AEBを無効化して走り切らせる

        if min_hazard_dist < 10.0:
            accel_cmd = -1.0  # フルブレーキ
            print(f"[AEB ACTIVE] Hazard detected! Estimated Distance: {min_hazard_dist:.2f}m. Full Braking.")
        elif min_hazard_dist < 18.0:
            accel_cmd = -0.3  # 警告減速
        else:
            # 通常速度追従 (PID)
            # CARLAでは前進速度はローカルのX軸成分を使うべきだが、ego_velはワールド座標の場合があるため速度ノルムを使用
            current_speed = np.linalg.norm(ego_vel) 
            error_v = target_v - current_speed
            accel_cmd = self.kp_long * error_v + self.kd_long * (error_v - self.prev_error_long) / dt
            self.prev_error_long = error_v
            accel_cmd = np.clip(accel_cmd, -1.0, 1.0)
            
        # 2-B. Lateral PID 回避操舵制御 (目標ライン追従)
        if self.demo_mode:
            error_y = offset_y - ego_pos[1]
        else:
            # 実機CARLAでの車線追従および横方向オフセット操舵制御
            carla_map = self.world.get_map()
            ego_loc_carla = self.ego_vehicle.get_location()
            waypoint = carla_map.get_waypoint(ego_loc_carla, project_to_road=True, lane_type=carla.LaneType.Driving)
            
            # 2.0m前方のウェイポイントを取得
            waypoints_ahead = waypoint.next(2.0)
            target_wp = waypoints_ahead[0] if waypoints_ahead else waypoint
            
            target_loc = target_wp.transform.location
            if offset_y != 0.0:
                right_vec = target_wp.transform.get_right_vector()
                target_loc += right_vec * offset_y
                
            # 目標座標を自車のローカル座標系に投影
            # 手动计算目标点在自车局部坐标系的坐标（替代 get_inverse().transform()）
            dx = target_loc.x - ego_transform.location.x
            dy = target_loc.y - ego_transform.location.y
            dz = target_loc.z - ego_transform.location.z
            fwd = ego_transform.get_forward_vector()
            right = ego_transform.get_right_vector()
            up = ego_transform.get_up_vector()
            local_x = dx * fwd.x + dy * fwd.y + dz * fwd.z
            local_y = dx * right.x + dy * right.y + dz * right.z
            local_z = dx * up.x + dy * up.y + dz * up.z
            target_local = carla.Location(x=local_x, y=local_y, z=local_z)
            error_y = target_local.y
            
        self.integral_error_lat += error_y * dt
        steer_cmd = (self.kp_lateral * error_y + 
                     self.kd_lateral * (error_y - self.prev_error_lat) / dt +
                     self.ki_lateral * self.integral_error_lat)
        self.prev_error_lat = error_y
        steer_cmd = np.clip(steer_cmd, -1.0, 1.0)
        
        # 3. 物理環境へ制御指令値を適用
        if self.demo_mode:
            self.mock_env.step([accel_cmd, steer_cmd])
        else:
            control = carla.VehicleControl()
            if accel_cmd >= 0:
                control.throttle = float(accel_cmd)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(-accel_cmd)
            control.steer = float(steer_cmd)
            self.ego_vehicle.apply_control(control)

        # 4. 知覚リスク・物理リスク・パラメータ決定の計算
        r_perceived, r_gt, gap, info = self.risk_calculator.calculate_multi_risk(
            ego_pos, ego_vel, gt_obstacles, yolo_detections
        )
        
        per_params = info['worst_perceived_params']
        gt_params = info['worst_gt_params']
        
        if not hasattr(self, 'max_gap_this_run'):
            self.max_gap_this_run = -float('inf')
            self.worst_case_image = None
            self.worst_case_step = 0
            
        if gap > self.max_gap_this_run:
            self.max_gap_this_run = gap
            if image is not None:
                self.worst_case_image = image.copy()
            self.worst_case_step = self.time_step
        
        # 5. 時系列ロギング
        log_entry = {
            'step': self.time_step,
            'scenario_ticks': self.scenario_ticks,
            'ego_x': ego_pos[0],
            'ego_y': ego_pos[1],
            'ego_vx': ego_vel[0],
            'worst_obstacle': info['worst_obstacle'],
            'worst_gt_distance': info['worst_gt_distance'],
            'worst_yolo_distance': info['worst_yolo_distance'],
            'r_perceived': r_perceived,
            'r_gt': r_gt,
            'perception_gap': gap,
            'mu_perceived': per_params.get('mu', 1.0),
            'mu_gt': gt_params.get('mu', 1.0),
            'omega_perceived': per_params.get('omega', 0.0),
            'omega_gt': gt_params.get('omega', 0.0),
            'alpha_perceived': per_params.get('alpha', 1.0),
            'alpha_gt': gt_params.get('alpha', 1.0),
            'beta_perceived': per_params.get('beta', 1.0),
            'beta_gt': gt_params.get('beta', 1.0),
            'offset_y': offset_y,
            'steer': steer_cmd,
            'accel': accel_cmd
        }
        
        # 5.5. 最小接近距離のトラッキングと衝突判定
        if len(gt_obstacles) > 0:
            current_min_dist = float('inf')
            for gt in gt_obstacles:
                if gt['class'] != 'unknown':
                    dist = np.linalg.norm(np.array(gt['pos'], dtype=float) - np.array(ego_pos, dtype=float))
                    if dist < current_min_dist:
                        current_min_dist = dist
                        
            if current_min_dist < self.scenario_min_distance:
                self.scenario_min_distance = current_min_dist
                
            if current_min_dist < 1.0:
                if getattr(self, '_collision_registered_this_scenario', False) == False:
                    self.scenario_collisions += 1
                    self._collision_registered_this_scenario = True
                    print(f"[CRASH] Collision detected! Min Distance: {current_min_dist:.2f}m")
                    
        log_entry['min_gt_distance'] = self.scenario_min_distance
        self.log_data.append(log_entry)
        
        # 6. 学習データの自動収集
        if hasattr(self, 'training_data') and len(gt_obstacles) > 0:
            for det in yolo_detections:
                det_class = det['class']
                if det_class in ['car', 'pedestrian', 'construction_signal', 'traffic_light']:
                    best_gt = None
                    min_dist_diff = float('inf')
                    for gt in gt_obstacles:
                        if gt['class'] == det_class:
                            d_gt = np.linalg.norm(np.array(gt['pos'], dtype=float) - np.array(ego_pos, dtype=float))
                            diff = abs(det['z_distance'] - d_gt)
                            if diff < min_dist_diff and diff < 10.0:
                                min_dist_diff = diff
                                best_gt = d_gt
                                
                    if best_gt is not None:
                        y_bot = det.get('bbox_y_bottom', 0.0)
                        h_norm = det.get('bbox_height', 0.0)
                        w_norm = det.get('bbox_width', 0.0)
                        
                        if y_bot > 0.0 and h_norm > 0.0 and w_norm > 0.0:
                            is_ped = 1 if det_class == 'pedestrian' else 0
                            is_car = 1 if det_class == 'car' else 0
                            is_sig = 1 if det_class == 'construction_signal' else 0
                            is_tl = 1 if det_class == 'traffic_light' else 0
                            
                            self.training_data.append({
                                'is_pedestrian': is_ped,
                                'is_car': is_car,
                                'is_construction_signal': is_sig,
                                'is_traffic_light': is_tl,
                                'bbox_y_bottom': y_bot,
                                'bbox_height': h_norm,
                                'bbox_width': w_norm,
                                'ego_speed': ego_vel[0],
                                'z_gt': best_gt
                            })
        
        # 7. シナリオ完了クリア判定
        if scenario_name == 'A':
            if len(gt_obstacles) > 0:
                p_pos = gt_obstacles[0]['pos']
                if p_pos[1] <= -2.5 or travel_dist >= 32.0:
                    self.clear_flag = True
            else:
                self.clear_flag = True
                
        elif scenario_name == 'B':
            if self.scenario_ticks > 80 and ego_vel[0] < 0.1:
                self.clear_flag = True
            elif self.scenario_ticks > 180:
                self.clear_flag = True
                
        elif scenario_name == 'C':
            if travel_dist >= 25.0:
                self.clear_flag = True
                
        elif scenario_name == 'D':
            if travel_dist >= 38.0:
                self.clear_flag = True
                
        elif scenario_name == 'E':
            if travel_dist >= 38.0:
                self.clear_flag = True

        if not self.demo_mode:
            self.next_frame_id = self.world.tick()
        
        # 每10帧更新一次旁观者视角
        if not self.demo_mode and self.time_step % 10 == 0:
            self._update_spectator()

        return log_entry
        
    def get_min_distance(self):
        return self.scenario_min_distance
        
    def get_collision_count(self):
        return self.scenario_collisions
        
    def get_worst_image(self):
        return getattr(self, 'worst_case_image', None)
        
    def get_worst_step(self):
        return getattr(self, 'worst_case_step', 0)
        
    def get_worst_case_parameters(self):
        step = getattr(self, 'worst_case_step', 0)
        for entry in self.log_data:
            if entry['step'] == step:
                return {
                    'mu_perceived': entry.get('mu_perceived', 1.0),
                    'mu_gt': entry.get('mu_gt', 1.0),
                    'omega_perceived': entry.get('omega_perceived', 0.0),
                    'omega_gt': entry.get('omega_gt', 0.0),
                    'alpha_perceived': entry.get('alpha_perceived', 1.0),
                    'alpha_gt': entry.get('alpha_gt', 1.0),
                    'beta_perceived': entry.get('beta_perceived', 1.0),
                    'beta_gt': entry.get('beta_gt', 1.0),
                    'worst_obstacle': entry.get('worst_obstacle', 'unknown')
                }
        return {}

    def run_experiment(self, scenario_name, target_speed_kph=40.0, gap=12.0, deceleration=-6.0, max_ticks=200):
        """
        指定された単一シナリオを実行。
        """
        print(f"\n===== Starting Scenario {scenario_name} =====")
        if self.demo_mode:
            self._setup_mock_scenario(scenario_name, target_speed_kph, gap, deceleration)
        else:
            self._setup_real_scenario(scenario_name, target_speed_kph, gap, deceleration)
            
        for tick in range(max_ticks):
            log = self.run_step(scenario_name, target_speed_kph)
            
            if tick % 10 == 0:
                print(f"Step {log['scenario_ticks']}: Ego X={log['ego_x']:.1f}m, Y={log['ego_y']:.2f}m, Vx={log['ego_vx']*3.6:.1f}km/h | worst={log['worst_obstacle']} | Gap={log['perception_gap']:.2f} (GT={log['r_gt']:.2f}, Perc={log['r_perceived']:.2f}) | mu_perc={log['mu_perceived']:.1f}, mu_gt={log['mu_gt']:.1f}")
                
            if self.clear_flag:
                print(f"--> Scenario {scenario_name} CLEARED at Step {tick}!")
                break
                
        if not self.demo_mode:
            self._destroy_actors()

    def run_sequence(self):
        """
        全シナリオ(A -> D -> B -> C)を切れ目なく連続走行。
        """
        print("\n===== Starting Scenario Sequence (Event-Driven Spawning) =====")
        
        current_seq = 'A'
        target_speed_kph = 40.0
        
        if self.demo_mode:
            self._setup_mock_scenario('A', target_speed_kph)
        else:
            self._setup_real_scenario('A', target_speed_kph)
            
        for tick in range(600):
            log = self.run_step(current_seq, target_speed_kph)
            
            if tick % 10 == 0:
                print(f"Seq [{current_seq}] Step {tick} (ScenTick {log['scenario_ticks']}): Ego X={log['ego_x']:.1f}m, Y={log['ego_y']:.2f}m | worst={log['worst_obstacle']} | Gap={log['perception_gap']:.2f} | mu_perc={log['mu_perceived']:.1f}, mu_gt={log['mu_gt']:.1f}")
            
            if self.clear_flag:
                ego_pos_x = log['ego_x']
                ego_vel_x = log['ego_vx']
                
                if current_seq == 'A':
                    print(f"--> Scenario A cleared! Spawning Scenario D (Barrier Avoidance) 25m ahead.")
                    current_seq = 'D'
                    target_speed_kph = 40.0
                    self.clear_flag = False
                    self.scenario_start_x = ego_pos_x
                    self.scenario_ticks = 0
                    
                    barrier_x = ego_pos_x + 25.0
                    if self.demo_mode:
                        self.mock_env.obstacles = [{
                            'class': 'construction_signal',
                            'pos': [barrier_x, 0.0, 0.0],
                            'vel': [0.0, 0.0, 0.0],
                            'mu': 1.5
                        }]
                    else:
                        if self.target_actor is not None:
                            self.target_actor.destroy()
                        barrier_bp = self.blueprint_library.find('static.prop.constructioncone')
                        ego_transform = self.ego_vehicle.get_transform()
                        barrier_loc = ego_transform.location + ego_transform.get_forward_vector() * 25.0
                        barrier_loc.z += 0.2
                        barrier_transform = carla.Transform(barrier_loc, ego_transform.rotation)
                        self.target_actor = self.world.spawn_actor(barrier_bp, barrier_transform)
                        self.actors.append(self.target_actor)
                        self.scenario_start_loc = ego_transform.location
                        self.scenario_start_x = ego_transform.location.x
                        
                elif current_seq == 'D':
                    print(f"--> Scenario D cleared! Spawning Scenario B (Lead Vehicle Sudden Brake) 20m ahead.")
                    current_seq = 'B'
                    target_speed_kph = 50.0
                    self.clear_flag = False
                    self.lead_decel_started = False
                    self.lead_decel_ticks = 40  # 新シナリオ開始から2秒後
                    self.lead_deceleration = -6.0
                    self.scenario_start_x = ego_pos_x
                    self.scenario_ticks = 0
                    
                    lead_x = ego_pos_x + 20.0
                    if self.demo_mode:
                        self.mock_env.obstacles = [{
                            'class': 'car',
                            'pos': [lead_x, 0.0, 0.0],
                            'vel': [ego_vel_x, 0.0, 0.0],
                            'mu': 1.0
                        }]
                    else:
                        if self.target_actor is not None:
                            self.target_actor.destroy()
                        lead_bp = self.blueprint_library.filter('model3')[0]
                        ego_transform = self.ego_vehicle.get_transform()
                        lead_loc = ego_transform.location + ego_transform.get_forward_vector() * 20.0
                        lead_loc.z += 0.5
                        lead_transform = carla.Transform(lead_loc, ego_transform.rotation)
                        self.target_actor = self.world.spawn_actor(lead_bp, lead_transform)
                        self.target_actor.set_target_velocity(carla.Vector3D(ego_vel_x, 0.0, 0.0))
                        self.actors.append(self.target_actor)
                        self.scenario_start_loc = ego_transform.location
                        self.scenario_start_x = ego_transform.location.x
                        
                elif current_seq == 'B':
                    print(f"--> Scenario B cleared! Spawning Scenario C (Crossing Vehicle) at oncoming path.")
                    current_seq = 'C'
                    target_speed_kph = 15.0
                    self.clear_flag = False
                    self.scenario_start_x = ego_pos_x
                    self.scenario_ticks = 0
                    
                    v_target = 11.11
                    intersect_x = ego_pos_x + 15.0
                    t_to_intersect = 15.0 / max(1.0, ego_vel_x)
                    target_start_x = intersect_x + v_target * t_to_intersect
                    
                    if self.demo_mode:
                        self.mock_env.obstacles = [{
                            'class': 'car',
                            'pos': [target_start_x, 3.5, 0.0],
                            'vel': [-v_target, 0.0, 0.0],
                            'mu': 1.0
                        }]
                    else:
                        if self.target_actor is not None:
                            self.target_actor.destroy()
                        lead_bp = self.blueprint_library.filter('model3')[0]
                        ego_transform = self.ego_vehicle.get_transform()
                        target_loc = ego_transform.location + ego_transform.get_forward_vector() * 15.0 + ego_transform.get_right_vector() * 3.5
                        target_loc.z += 0.5
                        target_rot = carla.Rotation(pitch=ego_transform.rotation.pitch,
                                                    yaw=ego_transform.rotation.yaw + 180.0,
                                                    roll=ego_transform.rotation.roll)
                        target_transform = carla.Transform(target_loc, target_rot)
                        self.target_actor = self.world.spawn_actor(lead_bp, target_transform)
                        self.actors.append(self.target_actor)
                        self.scenario_start_loc = ego_transform.location
                        self.scenario_start_x = ego_transform.location.x
                        
                elif current_seq == 'C':
                    print(f"--> Scenario C cleared! Spawning Scenario E (Red Light Intersection) 35m ahead.")
                    current_seq = 'E'
                    target_speed_kph = 40.0
                    self.clear_flag = False
                    self.scenario_start_x = ego_pos_x
                    self.scenario_ticks = 0
                    
                    target_x = ego_pos_x + 35.0
                    
                    if self.demo_mode:
                        self.mock_env.obstacles = [{
                            'class': 'traffic_light',
                            'pos': [target_x, 0.0, 0.0],
                            'vel': [0.0, 0.0, 0.0],
                            'color': 'red',
                            'mu': 2.0
                        }]
                    else:
                        if self.target_actor is not None:
                            self.target_actor.destroy()
                        barrier_bp = self.blueprint_library.find('static.prop.streetbarrier')
                        ego_transform = self.ego_vehicle.get_transform()
                        barrier_loc = ego_transform.location + ego_transform.get_forward_vector() * 35.0
                        barrier_loc.z += 2.0
                        barrier_transform = carla.Transform(barrier_loc, ego_transform.rotation)
                        self.target_actor = self.world.spawn_actor(barrier_bp, barrier_transform)
                        self.actors.append(self.target_actor)
                        self.scenario_start_loc = ego_transform.location
                        self.scenario_start_x = ego_transform.location.x
                        
                elif current_seq == 'E':
                    print(f"--> All Scenarios in sequence CLEARED!")
                    break
                    
        if not self.demo_mode:
            self._destroy_actors()

    def set_weather_params(self, sun_altitude, precipitation, fog_density):
        if self.demo_mode:
            self.mock_env.set_weather(sun_altitude, precipitation, fog_density)
        else:
            weather = carla.WeatherParameters(
                sun_altitude_angle=sun_altitude,
                precipitation=precipitation,
                fog_density=fog_density,
                cloudiness=max(precipitation, fog_density),
                wetness=precipitation,
                wind_intensity=10.0
            )
            self.world.set_weather(weather)
            frame_id = self.world.tick()
            if hasattr(self, 'next_frame_id'):
                self.next_frame_id = frame_id

    def visualize_and_save(self, save_path="results/risk_params_timeseries.png"):
        """
        時系列プロットを作成して保存。
        """
        if not self.log_data:
            print("[WARNING] No log data to plot.")
            return
            
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = pd.DataFrame(self.log_data)
        
        plt.figure(figsize=(14, 10))
        
        # グラフ1: リスク値の比較 (Perceived vs GT)
        plt.subplot(2, 1, 1)
        plt.plot(df['step'], df['r_gt'], label='Ground Truth Risk (R_gt)', color='green', linewidth=2)
        plt.plot(df['step'], df['r_perceived'], label='Perceived Risk (R_perceived)', color='blue', linestyle='--', linewidth=2)
        plt.fill_between(df['step'], df['r_gt'], df['r_perceived'], where=(df['r_gt'] > df['r_perceived']),
                         color='red', alpha=0.2, label='Perception Gap (Vulnerability)')
        plt.title("Time-Series Risk & Perception Gap", fontsize=14)
        plt.xlabel("Simulation Steps", fontsize=12)
        plt.ylabel("Risk Score", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=10)
        
        # グラフ2: 物体識別に基づくパラメータ決定挙動
        plt.subplot(2, 1, 2)
        plt.plot(df['step'], df['mu_gt'], label='GT Category Parameter (mu_gt)', color='forestgreen', linewidth=2)
        plt.plot(df['step'], df['mu_perceived'], label='Perceived Category Parameter (mu_perceived)', color='royalblue', linestyle='-.', linewidth=2)
        plt.plot(df['step'], df['omega_gt'], label='GT Interaction Weight (omega_gt)', color='orange', linewidth=1.5)
        plt.plot(df['step'], df['omega_perceived'], label='Perceived Interaction Weight (omega_perceived)', color='red', linestyle=':', linewidth=1.5)
        
        plt.title("Parameter Mapping from YOLO3D Object Classification", fontsize=14)
        plt.xlabel("Simulation Steps", fontsize=12)
        plt.ylabel("Parameter Value", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=10)
        
        # クラステキストラベル表示
        for i in range(0, len(df), max(1, len(df)//8)):
            row = df.iloc[i]
            if row['worst_obstacle'] != 'unknown':
                plt.text(row['step'], row['mu_perceived'], f"{row['worst_obstacle']}", 
                         fontsize=9, rotation=15, ha='right', va='bottom', color='black')
                         
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[SUCCESS] Timeseries validation plot saved at {save_path}")

    def get_max_gap(self):
        if not self.log_data:
            return 0.0
        df = pd.DataFrame(self.log_data)
        return float(df['perception_gap'].max())

    def shutdown(self):
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
            
        experiment.visualize_and_save(args.save_path)
        
    finally:
        experiment.shutdown()
