import os
import sys
import time
import argparse
import numpy as np

# Scenario cycling counter (A→B→C→D→E→A→...)
_scenario_cycle = 0
_SCENARIOS = ['A', 'B', 'C', 'D', 'E']
import cv2
import pandas as pd
import matplotlib.pyplot as plt



# リスク計算クラスとYOLO評価器の読み込みｿ
from risk_calculator import RiskCalculator
from evaluator import YoloEvaluator
from carla_mock import MockCarlaEnv

# CARLA縺ｮ繧､繝ｳ繝昴・繝郁ｩｦ陦・
try:
    import carla
    import queue
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False

class CameraOnlyExperiment:
    """
    繧ｫ繝｡繝ｩ蜊倅ｸ€繧ｻ繝ｳ繧ｵ繝ｼ・・amera-Only・峨↓繧医ｋ讓呎ｺ泡DAS繧ｷ繝翫Μ繧ｪ螳滄ｨ楢ｵｰ陦檎ｮ｡逅・け繝ｩ繧ｹ縲・    螳滓ｩ櫃ARLA縺翫ｈ縺ｳ繧ｪ繝輔Λ繧､繝ｳ繝｢繝・け・・-demo・峨・荳｡譁ｹ繧偵し繝昴・繝医＠縺ｾ縺吶€・    """
    def __init__(self, demo_mode=True, host='localhost', port=2000, record_video=False, run_id=None):
        self.demo_mode = demo_mode or not CARLA_AVAILABLE
        self.record_video = record_video
        self.video_frames = []
        self.evaluator = YoloEvaluator()
        
        self.risk_calculator = RiskCalculator()
        self.camera_type = 'RGB' # 繝・ヵ繧ｩ繝ｫ繝医・RGB
        self.actors = []
        self.training_data = []
        self.log_data = []
        
        self.run_id = run_id if run_id else time.strftime("%Y%m%d_%H%M%S")
        
        # HUD蜍慕判菫晏ｭ倡畑
        os.makedirs('results', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter('results/sensor_record.mp4', fourcc, 20.0, (1280, 720))
        self.current_scenario = None
        self.clear_flag = False
        self.lead_decel_ticks = float('inf')
        self.lead_decel_started = False
        self.lead_deceleration = -6.0
        
        self.scenario_min_distance = float('inf')
        self.scenario_collisions = 0
        
        # 繧ｷ繝翫Μ繧ｪ蝗ｺ譛峨・騾ｲ陦檎ｮ｡逅・､画焚
        self.scenario_start_x = 0.0
        self.scenario_start_loc = None  # CARLA逕ｨ縺ｮ髢句ｧ区凾 Location
        self.scenario_ticks = 0
        self.trigger_dist = 0.0         # 髢句ｧ倶ｽ咲ｽｮ縺九ｉ縺ｮ繝医Μ繧ｬ繝ｼ霍晞屬
        
        # PID繧ｳ繝ｳ繝医Ο繝ｼ繝ｩ逕ｨ繝代Λ繝｡繝ｼ繧ｿ・域ｨｪ譁ｹ蜷托ｼ・
        self.kp_lateral = 1.0
        self.kd_lateral = 0.2
        self.ki_lateral = 0.02
        self.integral_error_lat = 0.0
        self.prev_error_lat = 0.0
        
        # PID繧ｳ繝ｳ繝医Ο繝ｼ繝ｩ逕ｨ繝代Λ繝｡繝ｼ繧ｿ・育ｸｦ譁ｹ蜷托ｼ・        # PIDコントローラ用パラメータ（縦方向）
        self.kp_long = 0.8
        self.kd_long = 0.15
        self.prev_error_long = 0.0

        
        if self.demo_mode:
            print("[INFO] Running in MOCK DEMO mode (No CARLA server required).")
            base_img = "base_image.png"

            if not os.path.exists(base_img):
                # 繝€繝溘・逕ｻ蜒上ｒ菴懈・
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
            # Clean up all leftover actors from previous trials
            for actor in list(self.world.get_actors()):
                if actor.type_id.startswith(('vehicle.', 'walker.', 'sensor.', 'controller.')):
                    try:
                        actor.destroy()
                    except:
                        pass
            self.blueprint_library = self.world.get_blueprint_library()
            self.actors = []
            self.ego_vehicle = None
            self.camera = None
            self.target_actor = None
            self.image_queue = queue.Queue()
            
            # 蜷梧悄繝｢繝ｼ繝芽ｨｭ螳・
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_synchronous_mode(True)
            self.original_settings = self.world.get_settings()
            self.settings = self.world.get_settings()
            self.settings.synchronous_mode = True
            self.settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(self.settings)

    def _setup_mock_scenario(self, name, ego_speed_kph, gap=12.0, deceleration=-6.0):
        """
        繝｢繝・け迺ｰ蠅・↓縺翫￠繧句推繧ｷ繝翫Μ繧ｪ縺ｮ迚ｩ逅・ｪｨ譬ｼ・亥・譛滉ｽ咲ｽｮ繝ｻ騾溷ｺｦ繝ｻ霆碁％・峨ｒ蛻晄悄蛹悶＠縺ｾ縺吶€・        """
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
        
        # Ego蛻晄悄菴咲ｽｮ繝ｻ騾溷ｺｦ險ｭ螳・
        v_ego = (ego_speed_kph / 3.6)
        self.mock_env.ego_pos = [0.0, 0.0, 0.0]
        self.mock_env.ego_vel = [v_ego, 0.0, 0.0]
        
        # ターゲットアクター

        if name == 'A':
            # 繧ｷ繝翫Μ繧ｪA (CPNA: 豁ｩ陦瑚€・ｨｪ譁ｭ)
            v_ped = 1.39 # 5 km/h
            dist_cross = 30.0 # 讓ｪ譁ｭ菴咲ｽｮ X=30m
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
            # 繧ｷ繝翫Μ繧ｪB (CCRb: 蜈郁｡瑚ｻ頑€･蛻ｶ蜍・
            self.lead_decel_started = False
            self.lead_decel_ticks = 40 # 2遘貞ｾ・40ticks)縺ｫ諤･繝悶Ξ繝ｼ繧ｭ
            self.lead_deceleration = deceleration
            
            self.mock_env.obstacles = [{
                'class': 'car',
                'pos': [gap, 0.0, 0.0],
                'vel': [v_ego, 0.0, 0.0],
                'mu': 1.0
            }]
            print(f"[Scenario B Setup] Ego & Lead Speed: {ego_speed_kph} km/h. Gap: {gap}m. Braking after 2.0s with {deceleration} m/sﾂｲ")
            
        elif name == 'C':
            # 繧ｷ繝翫Μ繧ｪC (CCFtap: 莠､蟾ｮ霆贋ｸ｡)
            v_target = 11.11 # 40 km/h
            x_intersect = 20.0
            t_to_intersect = x_intersect / v_ego
            x_target_start = x_intersect + v_target * t_to_intersect
            
            self.mock_env.obstacles = [{
                'class': 'car',
                'pos': [x_target_start, 3.5, 0.0], # 蟇ｾ蜷題ｻ顔ｷ・Y=3.5
                'vel': [-v_target, 0.0, 0.0],
                'mu': 1.0
            }]
            print(f"[Scenario C Setup] Ego Speed: {ego_speed_kph} km/h. Oncoming starts at X={x_target_start:.2f}m with speed {v_target*3.6:.1f} km/h")
            
        elif name == 'D':
            # 繧ｷ繝翫Μ繧ｪD (AVOID: 髱咏噪髫懷ｮｳ迚ｩ蝗樣∩)
            self.mock_env.obstacles = [{
                'class': 'construction_signal',
                'pos': [35.0, 0.0, 0.0], # 霆顔ｷ壻ｸｭ螟ｮ X=35m
                'vel': [0.0, 0.0, 0.0],
                'mu': 1.5
            }]
            print(f"[Scenario D Setup] Ego Speed: {ego_speed_kph} km/h. Construction barrier at X=35m")
            
        elif name == 'E':
            # 繧ｷ繝翫Μ繧ｪE (RLI: 襍､菫｡蜿ｷ莠､蟾ｮ轤ｹ)
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
        螳滓ｩ櫃ARLA繧ｷ繝溘Η繝ｬ繝ｼ繧ｿ縺ｫ縺翫￠繧句推繧ｷ繝翫Μ繧ｪ縺ｮ迚ｩ逅・ｪｨ譬ｼ・医い繧ｯ繧ｿ繝ｼ驟咲ｽｮ縲・€溷ｺｦ蛻ｶ蠕｡蛻晄悄蛹厄ｼ・        """
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
        
        # Find a spawn point with a traffic light within 40m ahead (all scenarios)
        best_spawn = spawn_points[0]
        for sp in spawn_points:
            wp = self.world.get_map().get_waypoint(sp.location)
            found = False
            for _ in range(20):  # walk 40m ahead
                next_wps = wp.next(2.0)
                if not next_wps:
                    break
                wp = next_wps[0]
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
        
        # 1. 閾ｪ霆翫・繧ｹ繝昴・繝ｳ
        ego_bp = self.blueprint_library.filter('model3')[0]
        ego_transform.location.z += 0.5
        self.ego_vehicle = self.world.spawn_actor(ego_bp, ego_transform)
        self.actors.append(self.ego_vehicle)
        
        # 髢句ｧ倶ｽ咲ｽｮ縺ｮ險倬鹸
        self.scenario_start_loc = ego_transform.location
        self.scenario_start_x = ego_transform.location.x
        
        # 2. 繧ｹ繝・Ξ繧ｪ繧ｫ繝｡繝ｩ縺ｮ蜿悶ｊ莉倥￠ (Left & Right)
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '110')
        
        camera_transform_left = carla.Transform(
            carla.Location(x=2.0, y=-0.27, z=1.4),
            carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0)
        )
        self.camera_left = self.world.spawn_actor(camera_bp, camera_transform_left, attach_to=self.ego_vehicle)
        self.actors.append(self.camera_left)
        
        camera_transform_right = carla.Transform(
            carla.Location(x=2.0, y=0.27, z=1.4),
            carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0)
        )
        self.camera_right = self.world.spawn_actor(camera_bp, camera_transform_right, attach_to=self.ego_vehicle)
        self.actors.append(self.camera_right)
        
        # 繧ｭ繝･繝ｼ縺ｮ菴懈・縺ｨ繧ｯ繝ｪ繧｢
        import queue

        if not hasattr(self, 'image_queue_left'):
            self.image_queue_left = queue.Queue()
            self.image_queue_right = queue.Queue()
            
        while not self.image_queue_left.empty():
            self.image_queue_left.get()
        while not self.image_queue_right.empty():
            self.image_queue_right.get()
            
        # コールバックで(フレームID, 画像)を格納
        def _on_camera_capture_left(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            bgr_image = np.ascontiguousarray(array[:, :, :3])
            self.image_queue_left.put((image.frame, bgr_image))
            
        def _on_camera_capture_right(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            bgr_image = np.ascontiguousarray(array[:, :, :3])
            self.image_queue_right.put((image.frame, bgr_image))
            
        self.camera_left.listen(_on_camera_capture_left)
        self.camera_right.listen(_on_camera_capture_right)
        
        # --- 3. 繝輔Ν繧ｻ繝ｳ繧ｵ繝ｼ・・SC逕ｨ縺ｮ霑ｽ蜉繧ｻ繝ｳ繧ｵ繝ｼ ---
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '56000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('sensor_tick', '0.0')
        self.lidar = self.world.spawn_actor(lidar_bp, camera_transform_left, attach_to=self.ego_vehicle)
        self.actors.append(self.lidar)
        
        radar_bp = self.blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '30')
        radar_bp.set_attribute('range', '100')
        radar_bp.set_attribute('sensor_tick', '0.0')
        self.radar = self.world.spawn_actor(radar_bp, camera_transform_left, attach_to=self.ego_vehicle)
        self.actors.append(self.radar)
        
        depth_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '1280')
        depth_bp.set_attribute('image_size_y', '720')
        depth_bp.set_attribute('fov', '110')
        self.depth_cam = self.world.spawn_actor(depth_bp, camera_transform_left, attach_to=self.ego_vehicle)
        self.actors.append(self.depth_cam)
        
        seg_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', '1280')
        seg_bp.set_attribute('image_size_y', '720')
        seg_bp.set_attribute('fov', '110')
        self.seg_cam = self.world.spawn_actor(seg_bp, camera_transform_left, attach_to=self.ego_vehicle)
        self.actors.append(self.seg_cam)

        
        if not hasattr(self, 'lidar_queue'):
            self.lidar_queue = queue.Queue()
            self.radar_queue = queue.Queue()
            self.depth_queue = queue.Queue()
            self.seg_queue = queue.Queue()
            
        while not self.lidar_queue.empty(): self.lidar_queue.get()
        while not self.radar_queue.empty(): self.radar_queue.get()
        while not self.depth_queue.empty(): self.depth_queue.get()
        while not self.seg_queue.empty(): self.seg_queue.get()
        
        def _on_lidar(data): self.lidar_queue.put((data.frame, data))
        def _on_radar(data): self.radar_queue.put((data.frame, data))
        def _on_depth(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            self.depth_queue.put((image.frame, array))
        def _on_seg(image):
            image.convert(carla.ColorConverter.Raw) # Raw ID繧貞叙蠕・
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            self.seg_queue.put((image.frame, array))
            
        self.lidar.listen(_on_lidar)
        self.radar.listen(_on_radar)
        self.depth_cam.listen(_on_depth)
        self.seg_cam.listen(_on_seg)
        
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

        # 蛻晄悄鬥ｴ譟薙∪縺帷畑tick
        for _ in range(5):
            self.world.tick()

        # 鬥ｴ譟薙∪縺帷畑tick縺ｧ繧ｭ繝･繝ｼ縺ｫ貅懊∪縺｣縺溷商縺・判蜒上ｒ縺吶∋縺ｦ蜿悶ｊ蜃ｺ縺励※遐ｴ譽・☆繧・
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break
        
        # 譛€蛻昴・繧ｹ繝・ャ繝励・縺溘ａ縺ｫ縲∵怙蛻昴・ world.tick() 繧貞他繧薙〒縺翫￥
        self.next_frame_id = self.world.tick()
        
        # 譌∬ｧり€・ｧ・ｧ抵ｼ夊ｽｦ蟆ｾ蜷惹ｸ頑婿
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
        if hasattr(self, 'target_actor') and self.target_actor is not None and getattr(self.target_actor, 'is_alive', False):
            self.target_actor.destroy()
        self.target_actor = None
        if hasattr(self, 'ego_vehicle') and self.ego_vehicle is not None and getattr(self.ego_vehicle, 'is_alive', False):
            self.ego_vehicle.destroy()
        self.ego_vehicle = None
        self.camera = None
        print("[INFO] Cleaned up actors.")


    def set_weather_params(self, lighting_pattern_or_alt, precip, fog):
        if isinstance(lighting_pattern_or_alt, str):
            if lighting_pattern_or_alt == "Noon_FrontLight":
                sun_alt = 90.0
                sun_az = 0.0
            elif lighting_pattern_or_alt == "Evening_BackLight":
                sun_alt = 15.0
                sun_az = 180.0
            elif lighting_pattern_or_alt == "Night_Dark":
                sun_alt = -90.0
                sun_az = 0.0
            else:
                sun_alt = 45.0
                sun_az = 0.0
        else:
            sun_alt = float(lighting_pattern_or_alt)
            sun_az = 0.0

        self.current_weather = {
            'sun_altitude_angle': sun_alt,
            'sun_azimuth_angle': sun_az,
            'precipitation': precip,
            'fog_density': fog
        }

        if self.demo_mode:
            self.mock_env.sun_altitude_angle = sun_alt
            self.mock_env.precipitation = precip
            self.mock_env.fog_density = fog
        else:
            try:
                import carla
                weather = self.world.get_weather()
                weather.sun_altitude_angle = sun_alt
                weather.sun_azimuth_angle = sun_az
                weather.precipitation = precip
                weather.fog_density = fog
                weather.precipitation_deposits = precip
                weather.wind_intensity = 10.0
                weather.wetness = precip
                self.world.set_weather(weather)
            except Exception as e:
                print(f'[WARNING] Failed to set CARLA weather: {e}')

    def run_step(self, scenario_name, target_speed_kph):
        """
        1繧ｷ繝溘Η繝ｬ繝ｼ繧ｷ繝ｧ繝ｳ繧ｹ繝・ャ繝励ｒ螳溯｡後€ょ宛蠕｡縲∬ｪ崎ｭ假ｼ・OLO3D・峨€√Μ繧ｹ繧ｯ險育ｮ励€√Ο繧ｮ繝ｳ繧ｰ縲・        """
        dt = 0.05
        self.time_step += 1
        self.scenario_ticks += 1
        
        lidar_data = None
        radar_data = None
        measured_distance = None
        measured_radar_dist = None
        measured_lateral = 0.0
        measured_rel_vel = 0.0
        final_dist = None
        
        # 1. 迚ｩ逅・憾諷九♀繧医・繧ｫ繝｡繝ｩ逕ｻ蜒擾ｼ・OLO3D・峨・蜿門ｾ・
        if self.demo_mode:
            ego_pos = self.mock_env.ego_pos
            ego_vel = self.mock_env.ego_vel
            gt_data = self.mock_env.get_ground_truth()
            gt_obstacles = gt_data['obstacles']
            travel_dist = ego_pos[0] - self.scenario_start_x
            
            # 繧ｷ繝翫Μ繧ｪ蛻･繧｢繧ｯ繧ｿ繝ｼ遘ｻ蜍輔ヨ繝ｪ繧ｬ繝ｼ

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
                
            # 繝｢繝・け迺ｰ蠅・畑縺ｮ蜍慕噪讀懷・蝎ｨ縺ｮ讒狗ｯ・(YOLO縺ｮ螟ｩ蛟吶↓繧医ｋ隕玖誠縺ｨ縺励・霍晞屬隱､蟾ｮ縺ｮ蜀咲樟)
            sun_alt = self.mock_env.sun_altitude_angle
            fog = self.mock_env.fog_density
            precip = self.mock_env.precipitation
            
            image = self.mock_env.get_image()
            _ = self.evaluator.evaluate_multi(image, ego_speed=ego_vel[0])
            
            # 蟷ｾ菴募ｭｦ蠎ｧ讓吶↓蝓ｺ縺･縺阪€∝､ｩ蛟吶・蠖ｱ髻ｿ繧帝←逕ｨ縺励◆隱崎ｭ倡ｵ先棡繧堤函謌・
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
                                
                        # 蟷ｾ菴募ｭｦ逧・↓BBox繧ｵ繧､繧ｺ縺ｨ菴咲ｽｮ繧帝€・ｮ暦ｼ医Δ繝・け逕ｨ繝・・繧ｿ蜿朱寔縺ｮ縺溘ａ・・
                        
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
                        
                        # 謗･蝨ｰ轤ｹ Y2 縺ｮ險育ｮ・(H_cam = 1.4m)
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
            # 1. 譛溷ｾ・☆繧九ヵ繝ｬ繝ｼ繝ID (self.next_frame_id) 縺ｮ逕ｻ蜒上ｒ繧ｭ繝･繝ｼ縺九ｉ蜿門ｾ励☆繧具ｼ亥酔譛滂ｼ・
            image_left = None
            image_right = None
            lidar_data = None
            radar_data = None
            depth_img = None
            seg_img = None
            
            start_wait = time.time()
            while time.time() - start_wait < 2.0:
                try:
                    sensor_frame_l, bgr_img_l = self.image_queue_left.get(timeout=0.1)
                    sensor_frame_r, bgr_img_r = self.image_queue_right.get(timeout=0.1)
                    sensor_frame_lidar, l_data = self.lidar_queue.get(timeout=0.1)
                    sensor_frame_radar, r_data = self.radar_queue.get(timeout=0.1)
                    sensor_frame_depth, d_img = self.depth_queue.get(timeout=0.1)
                    sensor_frame_seg, s_img = self.seg_queue.get(timeout=0.1)
                    
                    frames = [sensor_frame_l, sensor_frame_r, sensor_frame_lidar, sensor_frame_radar, sensor_frame_depth, sensor_frame_seg]

                    if all(f == self.next_frame_id for f in frames):
                        image_left = bgr_img_l
                        image_right = bgr_img_r
                        lidar_data = l_data
                        radar_data = r_data
                        depth_img = d_img
                        seg_img = s_img
                        break
                    elif any(f < self.next_frame_id for f in frames):
                        # 蜿､縺・ヵ繝ｬ繝ｼ繝縺ｯ遐ｴ譽・
                        continue
                    else:
                        # 蜷梧悄繧ｺ繝ｬ
                        max_frame = max(frames)
                        print(f"[WARNING] Sync drift detected. Expected {self.next_frame_id}, got {max_frame}. Adjusting sync frame.")
                        self.next_frame_id = max_frame
                        image_left = bgr_img_l
                        image_right = bgr_img_r
                        lidar_data = l_data
                        radar_data = r_data
                        depth_img = d_img
                        seg_img = s_img
                        break
                except queue.Empty:
                    break

            
            if image_left is None:
                print(f"[WARNING] Mismatched or dropped sensor frame for tick {self.next_frame_id}. Perception will fall back.")
                image_left = np.zeros((720, 1280, 3), dtype=np.uint8)
                image_right = np.zeros((720, 1280, 3), dtype=np.uint8)
                depth_img = np.zeros((720, 1280, 4), dtype=np.uint8)
                seg_img = np.zeros((720, 1280, 4), dtype=np.uint8)
            else:
                # ユーザーの要望により、本番シミュレーション時に1度だけ左右のステレオ画像を保存
                if self.scenario_ticks == 50 and not self.demo_mode:
                    import os
                    os.makedirs('results', exist_ok=True)
                    cv2.imwrite('results/carla_stereo_left.jpg', image_left)
                    cv2.imwrite('results/carla_stereo_right.jpg', image_right)
                    print("[INFO] Saved a pair of raw stereo images from CARLA simulation to results/carla_stereo_left.jpg and right.jpg")
            
            # 2. 縺薙・繝輔Ξ繝ｼ繝縺ｫ縺翫￠繧区怙譁ｰ縺ｮ迚ｩ逅・憾諷九ｒ蜿門ｾ励☆繧・            # 縺薙ｌ縺ｫ繧医ｊ縲∝叙蠕励＠縺溽判蜒上→迚ｩ逅・憾諷九′螳悟・縺ｫ蜷後§繝輔Ξ繝ｼ繝ID (self.next_frame_id) 縺ｮ繧ゅ・縺ｧ蜷梧悄縺吶ｋ・・
            ego_transform = self.ego_vehicle.get_transform()
            ego_pos = [ego_transform.location.x, ego_transform.location.y, ego_transform.location.z]
            ego_vel_vec = self.ego_vehicle.get_velocity()
            ego_vel = [ego_vel_vec.x, ego_vel_vec.y, ego_vel_vec.z]
            travel_dist = ego_transform.location.distance(self.scenario_start_loc)

            
            if scenario_name == 'A':
                if self.target_actor is not None and travel_dist >= self.trigger_dist:
                    ped_control = carla.WalkerControl()
                    ped_control.direction = ego_transform.get_right_vector()
                    ped_control.speed = 1.39
                    self.target_actor.apply_control(ped_control)
            elif scenario_name == 'B':
                if self.target_actor is not None:
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
                    
                    # 霍晞屬縺瑚ｿ代▼縺・◆繧峨∬・辟ｶ縺ｪ迚ｩ逅・嫌蜍・VehicleControl)縺ｧ蠑ｷ蠑輔↓繝上Φ繝峨Ν繧貞・繧・
                    if dist_to_target < 20.0:
                        # steer蛟､繧帝←逕ｨ縺励※閾ｪ霆顔ｷ壼・(蟇ｾ蜷題ｻ翫°繧芽ｦ九※蟾ｦ譁ｹ蜷・縺ｫ蜑ｲ繧願ｾｼ繧
                        control = carla.VehicleControl(throttle=0.8, steer=-0.6)
                        self.target_actor.apply_control(control)
                    else:
                        # 驕縺・ｴ蜷医・逶ｴ騾ｲ
                        control = carla.VehicleControl(throttle=0.8, steer=0.0)
                        self.target_actor.apply_control(control)
            
            yolo_detections = self.evaluator.evaluate_multi(image_left, image_right=image_right, ego_speed=ego_vel[0])
            
            image = image_left.copy() if image_left is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
            
            gt_obstacles = []
            
            # --- Semantic LiDAR Processing ---
            global_dist_lidar = None
            if not self.demo_mode and lidar_data is not None:
                dtype = np.dtype([
                    ('x', np.float32), ('y', np.float32), ('z', np.float32),
                    ('cos_inc_angle', np.float32), ('object_idx', np.uint32), ('object_tag', np.uint32)
                ])
                lidar_array = np.frombuffer(lidar_data.raw_data, dtype=dtype)
                # Filter points matching Pedestrian (4) or Vehicles (10)
                valid_points = lidar_array[(lidar_array['object_tag'] == 4) | (lidar_array['object_tag'] == 10)]
                if len(valid_points) > 0:
                    global_dist_lidar = float(np.median(valid_points['x']))

            if radar_data is not None:
                radar_points = np.frombuffer(radar_data.raw_data, dtype=np.float32).reshape((-1, 4))
                front_radar = radar_points[np.abs(radar_points[:, 1]) < 0.1]

                if len(front_radar) > 0:
                    closest_idx = np.argmin(front_radar[:, 3])
                    measured_rel_vel = front_radar[closest_idx, 0]
                    measured_radar_dist = front_radar[closest_idx, 3]
                    
            v_ego_norm = np.linalg.norm(ego_vel)
            measured_target_vel = max(0.0, v_ego_norm - measured_rel_vel) if len(front_radar if radar_data is not None else []) > 0 else 0.0

            
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

        # カメラ認識のみに基づく走行制御
        closest_hazard = None
        min_hazard_dist = float('inf')
        offset_y = 0.0
        
        for det in yolo_detections:

            if det['yolo3d_rel_pos'] is not None:
                x_rel, y_rel, z_rel = det['yolo3d_rel_pos']
                
                # AVOID (蟾･莠狗畑繝舌Μ繧ｱ繝ｼ繝牙屓驕ｿ) 蛻､螳・
                if det['class'] == 'construction_signal' and z_rel <= 18.0:
                    offset_y = 2.5
                
                # Lane hazard: object in ego path (exclude traffic lights for non-E scenarios)
                is_lane_hazard = (abs(x_rel) < 1.8 
                                  and det['class'] != 'traffic_light')
                # Traffic light AEB: disabled (traffic lights never trigger AEB)
                is_red_light = False

                
                if (is_lane_hazard or is_red_light) and z_rel < min_hazard_dist:
                    min_hazard_dist = z_rel
                    closest_hazard = det
                    
        # 2-A. Longitudinal AEB 蛻ｶ蠕｡
        target_v = target_speed_kph / 3.6
        
        # 繝・・繧ｿ蜿朱寔逕ｨ縺ｪ縺ｩ縲、I縺梧悴蟄ｦ鄙偵〒繝輔か繝ｼ繝ｫ繝舌ャ繧ｯ縺瑚ｪ､菴懷虚縺吶ｋ蝣ｴ蜷医・繝悶Ξ繝ｼ繧ｭ繧堤┌隕悶☆繧・譛菴朱剞縺ｮ螳牙・陬・ｽｮ縺ｮ縺ｿ)
        # 霍晞屬謗ｨ螳哂I(Regressor)縺後Ο繝ｼ繝峨＆繧後※縺・↑縺・ｴ蜷医・縲∝ｼｷ蛻ｶ逧・↓PID霑ｽ蠕薙ｒ蜆ｪ蜈・
        if getattr(self.evaluator, 'distance_regressor', None) is None:
            min_hazard_dist = float('inf')  # AEB繧堤┌蜉ｹ蛹悶＠縺ｦ襍ｰ繧雁・繧峨○繧・

        if min_hazard_dist < 10.0:
            accel_cmd = -1.0  # Full Braking
            hazard_class = closest_hazard.get('class', 'unknown') if closest_hazard else 'unknown'
            print(f"[AEB ACTIVE] Scenario {scenario_name} | Hazard: {hazard_class} | Dist: {min_hazard_dist:.2f}m | Full Braking.")
        elif min_hazard_dist < 18.0:
            accel_cmd = -0.3  # 隴ｦ蜻頑ｸ幃€・
        
        else:
            # 騾壼ｸｸ騾溷ｺｦ霑ｽ蠕・(PID)
            # CARLA縺ｧ縺ｯ蜑埼€ｲ騾溷ｺｦ縺ｯ繝ｭ繝ｼ繧ｫ繝ｫ縺ｮX霆ｸ謌仙・繧剃ｽｿ縺・∋縺阪□縺後€‘go_vel縺ｯ繝ｯ繝ｼ繝ｫ繝牙ｺｧ讓吶・蝣ｴ蜷医′縺ゅｋ縺溘ａ騾溷ｺｦ繝弱Ν繝繧剃ｽｿ逕ｨ
            current_speed = np.linalg.norm(ego_vel) 
            error_v = target_v - current_speed
            accel_cmd = self.kp_long * error_v + self.kd_long * (error_v - self.prev_error_long) / dt
            self.prev_error_long = error_v
            accel_cmd = np.clip(accel_cmd, -1.0, 1.0)
            
        # 2-B. Lateral PID 蝗樣∩謫崎扱蛻ｶ蠕｡ (逶ｮ讓吶Λ繧､繝ｳ霑ｽ蠕・

        if self.demo_mode:
            error_y = offset_y - ego_pos[1]
        else:
            # 螳滓ｩ櫃ARLA縺ｧ縺ｮ霆顔ｷ夊ｿｽ蠕薙♀繧医・讓ｪ譁ｹ蜷代が繝輔そ繝・ヨ謫崎扱蛻ｶ蠕｡
            carla_map = self.world.get_map()
            ego_loc_carla = self.ego_vehicle.get_location()
            waypoint = carla_map.get_waypoint(ego_loc_carla, project_to_road=True, lane_type=carla.LaneType.Driving)
            
            # 2.0m蜑肴婿縺ｮ繧ｦ繧ｧ繧､繝昴う繝ｳ繝医ｒ蜿門ｾ・
            waypoints_ahead = waypoint.next(2.0)
            target_wp = waypoints_ahead[0] if waypoints_ahead else waypoint
            
            target_loc = target_wp.transform.location

            if offset_y != 0.0:
                right_vec = target_wp.transform.get_right_vector()
                target_loc += right_vec * offset_y
                
            # 逶ｮ讓吝ｺｧ讓吶ｒ閾ｪ霆翫・繝ｭ繝ｼ繧ｫ繝ｫ蠎ｧ讓咏ｳｻ縺ｫ謚募ｽｱ
            # 手动计算目标在自车局部坐标系中的坐标（替代 get_inverse().transform()）
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
        
        # 3. 迚ｩ逅・腸蠅・∈蛻ｶ蠕｡謖・ｻ､蛟､繧帝←逕ｨ

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

        # 4. 遨ｺ髢薙そ繝ｳ繧ｵ繝ｼ繝輔Η繝ｼ繧ｸ繝ｧ繝ｳ縺ｨ繝ｪ繧ｹ繧ｯ縺ｮ險育ｮ・
        measured_class = 'unknown'
        fused_dist = float('inf')
        global_dist_stereo = float('inf')
        global_dist_ai = float('inf')
        
        if not self.demo_mode:
            # 1. LiDAR距離を最優先で使用
            if global_dist_lidar is not None:
                fused_dist = global_dist_lidar
                measured_class = 'obstacle' # LiDAR単体ではクラス判別が難しいので汎用タグ
        else:
            global_dist_lidar = None
        
        # 2. YOLOからの各カメラ距離抽出
        for det in yolo_detections:
            stereo_d = det.get('z_distance', float('inf'))
            actual_stereo = det.get('dist_stereo', float('inf'))
            actual_ai = det.get('dist_ai', float('inf'))
            
            if 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                # 描画 (HUD)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f"{det['class']} C:{stereo_d:.1f}"
                cv2.putText(image, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # YOLOがLiDARより近い障害物を見つけた場合や、LiDARが欠損している場合はカメラ情報を信用
                if stereo_d < fused_dist:
                    fused_dist = stereo_d
                    measured_class = det['class']
                
                # 最も近いオブジェクトの情報を記録
                if actual_stereo < global_dist_stereo:
                    global_dist_stereo = actual_stereo
                if actual_ai < global_dist_ai:
                    global_dist_ai = actual_ai

        dist_for_risk = fused_dist if fused_dist != float('inf') else 100.0
        
        r_fusion, info = self.risk_calculator.calculate_fusion_risk(
            measured_distance=dist_for_risk,
            measured_rel_vel=measured_rel_vel if 'measured_rel_vel' in locals() else 0.0,
            measured_class=measured_class,
            lateral_offset=measured_lateral if 'measured_lateral' in locals() else 0.0
        )
        
        # 蠕梧婿莠呈鋤諤ｧ縺ｮ縺溘ａ縲∽ｻ･蜑阪・calculate_multi_risk繧ゆｸｦ陦後＠縺ｦ蜻ｼ縺ｳ蜃ｺ縺・
        r_perceived, r_gt, gap_multi, multi_info = self.risk_calculator.calculate_multi_risk(
            ego_pos, ego_vel, gt_obstacles, yolo_detections
        )
        
        # Optuna譛€驕ｩ蛹悶・逶ｮ逧・未謨ｰ縺ｨ縺励※縺ｮ Gap 縺ｯ莉･蜑阪・繧ゅ・繧偵◎縺ｮ縺ｾ縺ｾ蠑輔″邯吶＄・郁ｧ｣譫蝉ｺ呈鋤諤ｧ・・
        gap = gap_multi

        
        if not hasattr(self, 'max_gap_this_run'):
            self.max_gap_this_run = -float('inf')
            self.worst_case_image = None
            self.worst_case_step = 0

            
        if gap > self.max_gap_this_run:
            self.max_gap_this_run = gap

            if image is not None:
                self.worst_case_image = image.copy()
            self.worst_case_step = self.time_step

        if self.record_video and image is not None:
            self.video_frames.append(image.copy())
        
        # GT霍晞屬縺ｮ險育ｮ・
        dist_gt = -1.0

        if not self.demo_mode and hasattr(self, 'ego_vehicle') and self.ego_vehicle is not None and self.target_actor is not None and self.target_actor.is_alive:
            dist_gt = self.ego_vehicle.get_transform().location.distance(self.target_actor.get_transform().location)
        elif self.demo_mode and len(self.mock_env.obstacles) > 0:
            dist_gt = np.linalg.norm(np.array(self.mock_env.obstacles[0]['pos'], dtype=float) - np.array(ego_pos, dtype=float))
            
        # 5. 譎らｳｻ蛻励Ο繧ｮ繝ｳ繧ｰ
        log_entry = {
            'step': self.time_step,
            'scenario_ticks': self.scenario_ticks,
            'tick': self.scenario_ticks,
            'precipitation': getattr(self, 'current_weather', {}).get('precipitation', 0.0),
            'fog': getattr(self, 'current_weather', {}).get('fog_density', 0.0),
            'sun_altitude_angle': getattr(self, 'current_weather', {}).get('sun_altitude_angle', 90.0),
            'scenario_type': getattr(self, 'current_scenario', 'unknown'),
            'ego_x': ego_pos[0],
            'ego_y': ego_pos[1],
            'ego_vx': ego_vel[0],
            'worst_obstacle': measured_class,
            'fusion_distance': dist_for_risk,
            'dist_lidar': global_dist_lidar,
            'dist_stereo': global_dist_stereo,
            'dist_ai': global_dist_ai,
            'dist_gt': dist_gt,
            'v_approach': measured_rel_vel,
            'r_fusion': r_fusion,
            'r_perceived': r_perceived,
            'r_gt': r_gt,
            'perception_gap': gap,
            'mu': info.get('mu', 1.0),
            'mu_perceived': multi_info.get('mu_perceived', 1.0),
            'mu_gt': multi_info.get('mu_gt', 1.0),
            'omega': info.get('omega', 0.0),
            'omega_gt': multi_info.get('omega_gt', 0.0),
            'omega_perceived': multi_info.get('omega_perceived', 0.0),
            'alpha': info.get('alpha', 1.0),
            'beta': info.get('beta', 1.0),
            'worst_gt_distance': multi_info.get('worst_gt_distance', 0.0),
            'worst_yolo_distance': multi_info.get('worst_yolo_distance', float('inf')),
            'offset_y': error_y if 'error_y' in locals() else 0.0,
            'steer': steer_cmd,
            'accel': accel_cmd
        }
        
        # 5.5. 譛蟆乗磁霑題ｷ晞屬縺ｮ繝医Λ繝・く繝ｳ繧ｰ縺ｨ陦晉ｪ∝愛螳・
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
                    print(f"[CRASH] Scenario {scenario_name} | Collision! Min Dist: {current_min_dist:.2f}m")
                    
        log_entry['min_gt_distance'] = self.scenario_min_distance
        # HUDのテキスト描画と動画保存
        cv2.putText(image, f"Risk: {r_fusion:.2f} | Gap: {gap:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if r_fusion > 1.0 else (0, 255, 0), 2)
        cv2.putText(image, f"Action: {'Brake' if accel_cmd < 0 else 'Cruise'} | Steer: {steer_cmd:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        lidar_str = f"{global_dist_lidar:.1f}" if global_dist_lidar is not None else "None"
        cam_str = f"{global_dist_stereo:.1f}" if global_dist_stereo != float('inf') else "inf"
        cv2.putText(image, f"Dist (L/S/F): {lidar_str} / {cam_str} / {fused_dist:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        
        if hasattr(self, 'video_writer'):
            self.video_writer.write(image)
        
        self.log_data.append(log_entry)
        
        # 6. 蟄ｦ鄙偵ョ繝ｼ繧ｿ縺ｮ閾ｪ蜍募庶髮・
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
        
        # 7. 繧ｷ繝翫Μ繧ｪ螳御ｺ・け繝ｪ繧｢蛻､螳・
        if scenario_name == 'A':
            if self.scenario_ticks > 180:
                self.clear_flag = True
            elif len(gt_obstacles) > 0:
                p_pos = gt_obstacles[0]['pos']

                if travel_dist >= 32.0:
                    self.clear_flag = True
                else:
                    if getattr(self, 'demo_mode', False):
                        if p_pos[1] <= -2.5:
                            self.clear_flag = True
                    elif hasattr(self, 'ego_vehicle') and self.ego_vehicle is not None:
                        ego_tf = self.ego_vehicle.get_transform()
                        right_vec = ego_tf.get_right_vector()
                        dx = p_pos[0] - ego_pos[0]
                        dy = p_pos[1] - ego_pos[1]
                        rel_y = dx * right_vec.x + dy * right_vec.y
                        if rel_y >= 2.5:
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
        
        # 豈・0蟶ｧ譖ｴ譁ｰ荳谺｡譌∬ｧり・ｧ・ｧ・
        if not self.demo_mode and self.time_step % 10 == 0:
            self._update_spectator()

        return log_entry
        
    def get_min_distance(self):
        return self.scenario_min_distance
        
    def get_collision_count(self):
        return self.scenario_collisions
        
    def get_worst_image(self):
        return getattr(self, 'worst_case_image', None)
        
    def _save_worst_image(self, scenario_name):
        """Save worst-case edge-case image to disk."""
        img = self.get_worst_image()
        if img is not None:
            step = getattr(self, 'worst_case_step', 0)
            filename = f"results/worst_case_{scenario_name}_step{step}.jpg"
            cv2.imwrite(filename, img)
            print(f"[SAVE] Worst-case edge-case image saved: {filename}")
        
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
        謖・ｮ壹＆繧後◆蜊倅ｸ繧ｷ繝翫Μ繧ｪ繧貞ｮ溯｡後・        """
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

        self._save_worst_image(scenario_name)


        if not self.demo_mode:
            self._destroy_actors()

    def _safe_spawn_actor(self, bp, target_loc, rot, forward_vector):
        import carla
        loc = carla.Location(target_loc.x, target_loc.y, target_loc.z)
        
        for i in range(5):
            ray_start = loc + carla.Location(z=1.0)
            ray_end = loc - carla.Location(z=1.0)
            labeled_objects = self.world.cast_ray(ray_start, ray_end)
            
            actor = None
            if len(labeled_objects) == 0:
                actor = self.world.try_spawn_actor(bp, carla.Transform(loc, rot))
                
            if actor is not None:
                return actor
                
            print(f"⚠️ Obstacle detected at spawn location. Adjusting backward... (Attempt {i+1}/5)")
            loc -= forward_vector * 1.5
            
        print("⚠️ All spawn attempts failed. Using fallback to road center.")
        wp = self.world.get_map().get_waypoint(target_loc)
        if wp:
            fallback_loc = wp.transform.location
            fallback_loc.z += 1.0
            return self.world.try_spawn_actor(bp, carla.Transform(fallback_loc, rot))
        return None

    def _inject_dynamic_hazard(self, seq, spawn_dist):
        if self.demo_mode:
            return
            
        try:
            import carla
        except ImportError:
            return
            
        ego_transform = self.ego_vehicle.get_transform()
        wp = self.world.get_map().get_waypoint(ego_transform.location)
        target_wps = wp.next(spawn_dist)
        target_wp = target_wps[0] if target_wps else wp
        
        base_loc = target_wp.transform.location
        forward = target_wp.transform.get_forward_vector()
        right = target_wp.transform.get_right_vector()

        # Set traffic light to green for A, B, C, D scenarios
        if seq in ['A', 'B', 'C', 'D']:
            try:
                import carla
                for tl in self.world.get_actors().filter('traffic.traffic_light'):
                    if tl.get_location().distance(target_wp.transform.location) < 20.0:
                        tl.set_state(carla.TrafficLightState.Green)
                        tl.freeze(True)
                        break
            except:
                pass

        if seq == 'A':
            bp_list = self.blueprint_library.filter('walker.pedestrian.0001')
            bp = bp_list[0] if len(bp_list) > 0 else self.blueprint_library.filter('walker.*')[0]
            # Place pedestrian directly in front of ego, offset right 3m
            fwd = ego_transform.get_forward_vector()
            ego_right = ego_transform.get_right_vector()
            loc = ego_transform.location + fwd * 30.0 - ego_right * 3.0
            loc.z += 0.5
            rot = carla.Rotation(yaw=ego_transform.rotation.yaw + 90.0)
            self.target_actor = self._safe_spawn_actor(bp, loc, rot, fwd)
            self.trigger_dist = 30.0 - (1.39 * 3.0 / 1.39)  # ~27m trigger for walker crossing

            if self.target_actor:
                self.actors.append(self.target_actor)
                control = carla.WalkerControl()
                control.direction = ego_right
                control.speed = 1.39
                self.target_actor.apply_control(control)
                
        elif seq == 'B':
            bp = self.blueprint_library.filter('model3')[0]
            fwd = ego_transform.get_forward_vector()
            loc = ego_transform.location + fwd * 30.0
            loc.z += 1.0
            rot = ego_transform.rotation
            self.target_actor = self._safe_spawn_actor(bp, loc, rot, fwd)

            if self.target_actor:
                self.actors.append(self.target_actor)
                self.target_actor.set_target_velocity(self.ego_vehicle.get_velocity())
                
        elif seq == 'C':
            bp = self.blueprint_library.filter('model3')[0]
            # Same as E: 35m ahead + 6m right, facing left across road
            fwd = ego_transform.get_forward_vector()
            ego_right = ego_transform.get_right_vector()
            loc = ego_transform.location + fwd * 35.0 + ego_right * 6.0
            loc.z += 1.0
            rot = carla.Rotation(pitch=0.0, yaw=ego_transform.rotation.yaw - 90.0, roll=0.0)
            self.target_actor = self._safe_spawn_actor(bp, loc, rot, fwd)

            if self.target_actor:
                self.actors.append(self.target_actor)
                # Steer hard left to cut fully across ego's lane
                control = carla.VehicleControl()
                control.throttle = 1.0
                control.steer = -1.0
                self.target_actor.apply_control(control)
                
        elif seq == 'D':
            bp_list = self.blueprint_library.filter('static.prop.constructioncone')
            bp = bp_list[0] if len(bp_list) > 0 else self.blueprint_library.filter('static.prop.*')[0]
            # Place directly in front of ego on its forward path
            fwd = ego_transform.get_forward_vector()
            loc = ego_transform.location + fwd * 35.0
            loc.z += 1.0
            rot = ego_transform.rotation
            self.target_actor = self._safe_spawn_actor(bp, loc, rot, forward)

            if self.target_actor:
                self.actors.append(self.target_actor)
                
        elif seq == 'E':
            bp_list = self.blueprint_library.filter('vehicle.ford.mustang')
            bp = bp_list[0] if len(bp_list) > 0 else self.blueprint_library.filter('vehicle.*')[0]
            # Spawn on right side road at junction ahead, facing towards ego's lane
            fwd = ego_transform.get_forward_vector()
            ego_right = ego_transform.get_right_vector()
            loc = ego_transform.location + fwd * 35.0 + ego_right * 6.0
            loc.z += 1.0
            # Face towards ego's lane (perpendicular, turning left from right)
            rot = carla.Rotation(pitch=0.0, yaw=ego_transform.rotation.yaw - 90.0, roll=0.0)
            self.target_actor = self._safe_spawn_actor(bp, loc, rot, fwd)

            if self.target_actor:
                self.actors.append(self.target_actor)
                control = carla.VehicleControl()
                control.throttle = 1.0
                self.target_actor.apply_control(control)

    def run_sequence(self):
        global _scenario_cycle
        import random
        import numpy as np
        target_speed_kph = 40.0
        consecutive_stopped_ticks = 0

        seq = _SCENARIOS[_scenario_cycle % 5]
        _scenario_cycle += 1
        spawn_dist = random.uniform(30.0, 40.0)

        if self.demo_mode:
            self._setup_mock_scenario(seq, target_speed_kph)
            self.current_scenario = seq
            for _ in range(40):
                self.run_step(seq, target_speed_kph)
            self.trigger_dist = self.scenario_start_x + (self.mock_env.ego_pos[0] - self.scenario_start_x) - 0.1
            for tick in range(100):
                self.run_step(seq, target_speed_kph)
        else:
            self._setup_real_scenario('sequence', target_speed_kph)
            self.current_scenario = seq
            
            for _ in range(40):
                self.run_step('sequence', target_speed_kph)
                
            # Delayed spawn for C (wait 10 ticks before appearing)
            if seq != 'C':
                self._inject_dynamic_hazard(seq, spawn_dist)
            
            print(f"\n===== [Trial Start] Scenario {seq} | Phase 0 | Weather: sun={getattr(self,'current_weather',{}).get('sun_altitude_angle',90):.1f}deg rain={getattr(self,'current_weather',{}).get('precipitation',0):.1f} fog={getattr(self,'current_weather',{}).get('fog_density',0):.1f} =====")
            
            scenario_phase = 0
            tick = 0
            c_spawned = False  # delayed spawn flag for C
            
            while True:
                tick += 1
                
                # Delayed C spawn at tick 10
                if seq == 'C' and not c_spawned and tick == 10:
                    self._inject_dynamic_hazard(seq, spawn_dist)
                    c_spawned = True
                
                try:
                    import carla
                    if not self.demo_mode and hasattr(self, 'target_actor') and self.target_actor:
                        if seq == 'B' and scenario_phase == 0 and tick == 30:
                            control = carla.VehicleControl()
                            control.brake = 1.0
                            control.throttle = 0.0
                            self.target_actor.apply_control(control)

                except ImportError:
                    pass
                    
                log = self.run_step(seq, target_speed_kph)

                if hasattr(self, 'ego_vehicle') and self.ego_vehicle is not None:
                    import numpy as np
                    vel = self.ego_vehicle.get_velocity()
                    ego_speed = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                    
                    if tick > 15 and ego_speed < 0.5:
                        consecutive_stopped_ticks += 1
                    else:
                        consecutive_stopped_ticks = 0
                        
                    if consecutive_stopped_ticks >= 5:
                        print(f"[Tick {tick}] Scenario {seq} | Phase {scenario_phase} | Stopped for 5 ticks.")
                        if scenario_phase == 0:
                            if hasattr(self, 'target_actor') and self.target_actor:
                                print(f"[Tick {tick}] Scenario {seq} | Destroying initial obstacle.")
                                self.target_actor.destroy()
                                self.target_actor = None
                            consecutive_stopped_ticks = 0
                            scenario_phase = 1
                        elif scenario_phase == 2:
                            print(f"[Tick {tick}] Scenario {seq} | FULL SEQUENCE SUCCESS.")
                            break
                            
                    if scenario_phase == 1 and ego_speed > 5.0:
                        print(f"[Tick {tick}] Scenario {seq} | Ego resumed. Spawning Phase 2 obstacle.")
                        try:
                            import carla
                            ego_tf = self.ego_vehicle.get_transform()
                            fwd = ego_tf.get_forward_vector()
                            ego_right = ego_tf.get_right_vector()
                            car_loc = ego_tf.location + fwd * 30.0
                            car_loc.z += 1.0
                            rot = ego_tf.rotation
                            
                            if seq == 'A':
                                bp = self.blueprint_library.filter('walker.pedestrian.0001')[0]
                                ped_loc = ego_tf.location + fwd * 30.0 - ego_right * 3.0
                                ped_loc.z += 0.5
                                ped_rot = carla.Rotation(yaw=ego_tf.rotation.yaw + 90.0)
                                self.target_actor = self._safe_spawn_actor(bp, ped_loc, ped_rot, fwd)
                                if self.target_actor:
                                    self.actors.append(self.target_actor)
                                    control = carla.WalkerControl()
                                    control.direction = ego_right
                                    control.speed = 1.39
                                    self.target_actor.apply_control(control)
                            elif seq == 'B':
                                bp = self.blueprint_library.filter('model3')[0]
                                self.target_actor = self._safe_spawn_actor(bp, car_loc, rot, fwd)
                                if self.target_actor:
                                    self.actors.append(self.target_actor)
                                    self.target_actor.set_target_velocity(self.ego_vehicle.get_velocity())
                                    self.lead_decel_ticks = 0
                                    self.lead_decel_started = True
                                    self.lead_deceleration = -9.0
                            elif seq == 'C':
                                bp = self.blueprint_library.filter('model3')[0]
                                c_loc = ego_tf.location + fwd * 35.0 + ego_right * 6.0
                                c_loc.z += 1.0
                                c_rot = carla.Rotation(yaw=ego_tf.rotation.yaw - 90.0)
                                self.target_actor = self._safe_spawn_actor(bp, c_loc, c_rot, fwd)
                                if self.target_actor:
                                    self.actors.append(self.target_actor)
                                    control = carla.VehicleControl()
                                    control.throttle = 1.0
                                    control.steer = -1.0
                                    self.target_actor.apply_control(control)
                            elif seq == 'D':
                                bp = self.blueprint_library.filter('static.prop.constructioncone')[0]
                                d_loc = ego_tf.location + fwd * 35.0
                                d_loc.z += 1.0
                                self.target_actor = self._safe_spawn_actor(bp, d_loc, rot, fwd)
                                if self.target_actor:
                                    self.actors.append(self.target_actor)
                            elif seq == 'E':
                                bp = self.blueprint_library.filter('vehicle.ford.mustang')[0]
                                e_loc = ego_tf.location + fwd * 35.0 + ego_right * 6.0
                                e_loc.z += 1.0
                                e_rot = carla.Rotation(yaw=ego_tf.rotation.yaw - 90.0)
                                self.target_actor = self._safe_spawn_actor(bp, e_loc, e_rot, fwd)
                                if self.target_actor:
                                    self.actors.append(self.target_actor)
                                    control = carla.VehicleControl()
                                    control.throttle = 1.0
                                    self.target_actor.apply_control(control)
                            
                            if self.target_actor:
                                scenario_phase = 2
                        except Exception as e:
                            print(f"Spawn Error at Phase 2: {e}")
                            
                if tick > 200:
                    print(f"[Timeout] Scenario {seq} | Forcing end of trial.")
                    break
                        
            self._save_worst_image(seq)

        
        if self.record_video and len(self.video_frames) > 0:
            import cv2
            import os
            import time
            os.makedirs("results/videos", exist_ok=True)
            h, w = self.video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = f"results/videos/sequence_{seq}_{int(time.time())}.mp4"
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
            for frame in self.video_frames:
                out.write(frame)
            out.release()
            print(f"[INFO] Saved sequence video to {video_path}")
            self.video_frames = []

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
            
        # シミュレーション終了時にCSVデータを自動エクスポート
        self.export_training_data()
        self.export_log_data()

    def get_max_gap(self):
        return getattr(self, 'max_gap_this_run', 0.0)
        
    def get_worst_case_image(self):
        return getattr(self, 'worst_case_image', None)

    def export_training_data(self, filepath=None):
        if filepath is None:
            filepath = f"results/distance_training_data_{self.run_id}.csv"
        if not self.training_data:
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame(self.training_data)
        # 複数トライアルで蓄積するため追記モード(a)で保存
        df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)
        print(f"[SUCCESS] Appended {len(df)} distance training samples to {filepath}")

    def export_log_data(self, filepath=None):
        if filepath is None:
            filepath = f"results/experiment_log_{self.run_id}.csv"
        if not self.log_data:
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame(self.log_data)
        # 複数トライアルで蓄積するため追記モード(a)で保存
        df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)
        print(f"[SUCCESS] Appended {len(df)} experiment log entries to {filepath}")


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
            
        # experiment.visualize_and_save(args.save_path, scenario_name=args.scenario)
        
    finally:
        experiment.shutdown()
