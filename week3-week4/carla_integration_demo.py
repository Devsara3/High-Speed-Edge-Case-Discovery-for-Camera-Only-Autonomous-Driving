# -*- coding: utf-8 -*-
"""
CARLA Integration Demo (統合メインループ & 制御チューニング用スクリプト)

貼り付けられたColabチュートリアルの実装（Step 1〜7）に完全に準拠しています。
Ego車両として Cybertruck をスポーンし、前方10mに静的障害物（Model 3）を配置、
チュートリアルで定義された各マウント位置・角度に6センサーを設置します。
"""

import time
import sys
import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

# ====== 自作モジュールのインポート ======
from carla_pid_controller import CarlaPIDController
from carla_sensor_manager import CarlaSensorManager
from carla_waypoint_planner import CarlaWaypointPlanner

try:
    import carla
except ImportError:
    print("【エラー】CARLAモジュールが見つかりません。")
    print("CARLAのPython環境（PythonAPI/carla/dist内のegg/whlファイル）が正しく設定されているか確認してください。")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("【エラー】OpenCV（cv2）が見つかりません。'pip install opencv-python' を実行してください。")
    sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="CARLA Autonomous Control & Sensor Suite Demo")
    
    # CARLA接続設定
    parser.add_argument('--host', type=str, default='localhost', help='CARLA Server IP Address (Default: localhost)')
    parser.add_argument('--port', type=int, default=2000, help='CARLA Server TCP Port (Default: 2000)')
    
    # 走行設定
    parser.add_argument('--target-speed', type=float, default=10.0, help='Target speed in m/s (Default: 10.0 m/s = 36 km/h)')
    
    # 縦方向 (速度) PIDゲイン
    parser.add_argument('--kp-lon', type=float, default=1.0, help='Longitudinal P Gain (Default: 1.0)')
    parser.add_argument('--ki-lon', type=float, default=0.1, help='Longitudinal I Gain (Default: 0.1)')
    parser.add_argument('--kd-lon', type=float, default=0.05, help='Longitudinal D Gain (Default: 0.05)')
    
    # 横方向 (ステアリング) PIDゲイン
    parser.add_argument('--kp-lat', type=float, default=0.8, help='Lateral P Gain (Default: 0.8)')
    parser.add_argument('--ki-lat', type=float, default=0.02, help='Lateral I Gain (Default: 0.02)')
    parser.add_argument('--kd-lat', type=float, default=0.1, help='Lateral D Gain (Default: 0.1)')
    
    # 記録の無効化オプション
    parser.add_argument('--no-record', action='store_true', help='Disable saving video and tuning plots upon exit')
    
    # オートパイロットテスト（自動ブレーキ・衝突回避）オプション
    parser.add_argument('--use-autopilot', action='store_true', help='Use CARLA Traffic Manager autopilot for Ego vehicle (tests collision avoidance / emergency braking)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("====================================================")
    print(" CARLA PID Control & Sensor Suite Integration System")
    print(f" Connecting to Server: {args.host}:{args.port}")
    print(f" Target Speed: {args.target_speed} m/s")
    print(f" Lon PID (Speed): Kp={args.kp_lon}, Ki={args.ki_lon}, Kd={args.kd_lon}")
    print(f" Lat PID (Steer): Kp={args.kp_lat}, Ki={args.ki_lat}, Kd={args.kd_lat}")
    print("====================================================")
            
    # --- [CARLAサーバー接続] ---
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()
        map_name = world.get_map().name
        print(f"Successfully connected to CARLA. Current Map: {map_name}")
    except Exception as e:
        print(f"【接続エラー】CARLAサーバーに接続できませんでした: {e}")
        print(f"CARLAシミュレータが起動していること、およびポート {args.port} が開放されていることを確認してください。")
        sys.exit(1)
        
    # シミュレータ同期モードの設定
    settings = world.get_settings()
    original_settings = world.get_settings()  # 終了時の復元用
    
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS (PIDのdtと一致させる)
    world.apply_settings(settings)
    
    vehicle = None
    obstacle_vehicle = None
    sensor_manager = None
    video_writer = None
    
    # 走行ログ記録用リスト
    log_time = []
    log_speed = []
    log_target_speed = []
    log_steer = []
    log_throttle = []
    log_brake = []
    log_cte = []
    log_heading_err = []
    
    # センサー可視化用のデータログ
    log_imu_t = []
    log_imu_accel = []  # [(ax, ay, az), ...]
    log_gnss_t = []
    log_gnss_coords = []  # [(lat, lon), ...]
    
    # 最新の点群データを保持（最終フレームの可視化用）
    latest_lidar_points = None
    latest_radar_points = None
    
    try:
        # --- [車両と障害物のスポーン] ---
        blueprint_library = world.get_blueprint_library()
        
        # 1. Ego車両のスポーン (Tesla Cybertruck: Tutorial Step 3に準拠)
        ego_bp = blueprint_library.find('vehicle.tesla.cybertruck')
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()
        vehicle = world.spawn_actor(ego_bp, spawn_point)
        print(f"Ego Vehicle spawned successfully (Tesla Cybertruck, ID: {vehicle.id}).")
        
        # 2. 静的障害物車両のスポーン (前方10m〜14mの位置にTesla Model 3を配置)
        # クライアント-サーバー同期のタイミングバグを回避するため、新規アクターの get_transform() ではなく
        # 既知の spawn_point から障害物のスポーン座標を計算します。
        obstacle_bp = blueprint_library.find('vehicle.tesla.model3')
        forward_vector = spawn_point.get_forward_vector()
        
        obstacle_vehicle = None
        # 道路やコリジョンとの競合によるスポーン失敗を防止するため、Z座標を少し浮かせ、
        # 失敗した場合は距離を離して最大3回まで再試行します。
        for attempt in range(3):
            spawn_dist = 10.0 + attempt * 2.0
            obs_location = spawn_point.location + forward_vector * spawn_dist
            obs_location.z += 0.3  # コリジョン判定を避けるため地面から少し浮かせる
            obstacle_transform = carla.Transform(obs_location, spawn_point.rotation)
            
            obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_transform)
            if obstacle_vehicle:
                print(f"Obstacle Vehicle spawned successfully {spawn_dist}m ahead (Tesla Model 3, ID: {obstacle_vehicle.id}).")
                obstacle_vehicle.set_autopilot(False)
                # 物理的な接地フェーズで接地させてから固定するため、ここでは物理演算はONのままにします。
                break
        
        if not obstacle_vehicle:
            print("【警告】障害物車両のスポーンに失敗しました（初期位置の競合の可能性があります）")
        
        # モジュール初期化
        sensor_manager = CarlaSensorManager(world, vehicle)
        planner = CarlaWaypointPlanner(world, vehicle)
        
        # コントローラー初期化
        controller = CarlaPIDController(vehicle, dt=0.05)
        controller.lon_kp, controller.lon_ki, controller.lon_kd = args.kp_lon, args.ki_lon, args.kd_lon
        controller.lat_kp, controller.lat_ki, controller.lat_kd = args.kp_lat, args.ki_lat, args.kd_lat
        
        # Ego車両の自動ブレーキ・衝突回避テスト用のオートパイロット切り替え
        if args.use_autopilot:
            print("Autopilot Test Mode: Enabling CARLA Traffic Manager for Ego Vehicle.")
            vehicle.set_autopilot(True)
            # トラフィックマネージャーのポート8000を取得して安全車間距離等を設定
            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_global_distance_to_leading_vehicle(4.0) # 安全車間距離を4.0mに設定
            # 衝突回避が正しく機能するように信号無視等を設定（直進テストをしやすくするため）
            traffic_manager.ignore_lights_percentage(vehicle, 100.0)
            traffic_manager.ignore_signs_percentage(vehicle, 100.0)
        else:
            print("Manual PID Control Mode: Tuning control loop.")
            vehicle.set_autopilot(False)
        
        # --- [各種センサーの設置 (Tutorial Step 4に完全準拠)] ---
        # 4.1 & 4.2 RGBカメラ & セマンティックカメラ（フロントガラス高さに設置）
        cam_location = carla.Location(x=1.5, y=0.0, z=1.7)
        cam_rotation = carla.Rotation(pitch=-10.0, yaw=0.0, roll=0.0)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        sensor_manager.spawn_rgb_camera(cam_transform, role_name='rgb_front')
        sensor_manager.spawn_semantic_segmentation_camera(cam_transform, role_name='seg_front')
        
        # 4.3 LiDAR（ルーフ中央）
        lidar_location = carla.Location(x=0.0, y=0.0, z=2.5)
        lidar_rotation = carla.Rotation(pitch=0.0)
        lidar_transform = carla.Transform(lidar_location, lidar_rotation)
        sensor_manager.spawn_lidar(lidar_transform, role_name='lidar')
        
        # 4.4 Radar（フロントバンパー）
        radar_location = carla.Location(x=2.0, y=0.0, z=0.5)
        radar_rotation = carla.Rotation(pitch=0.0, yaw=0.0)
        radar_transform = carla.Transform(radar_location, radar_rotation)
        sensor_manager.spawn_radar(radar_transform, role_name='radar')
        
        # 4.5 IMU（車体中心）
        imu_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        sensor_manager.spawn_imu(imu_transform, role_name='imu')
        
        # 4.6 GNSS (GPS)（ルーフ中央）
        gnss_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.5))
        sensor_manager.spawn_gnss(gnss_transform, role_name='gnss')
        
        print("All sensors initialized. Starting autonomous loop... (Press Ctrl+C to stop)")
        
        # センサーがデータを流し始めるまで同期クロックを進め、同時に物理的に路面に接地させる（初期接地フェーズ）
        print("Settle phase: Letting vehicles settle on the road surface...")
        for _ in range(15):
            # 自車は手動ブレーキで動かないように保持
            vehicle.apply_control(carla.VehicleControl(hand_brake=True))
            world.tick()
            
        # 障害物が接地した状態の座標で物理演算を無効化（固定）
        if obstacle_vehicle:
            obstacle_vehicle.set_simulate_physics(False)
            print("Obstacle vehicle physics frozen on the road surface.")
            
        step_count = 0
        consecutive_stopped = 0
        
        # ====== メイン制御ループ ======
        while True:
            # シミュレータの時計を1コマ(0.05秒)進める
            world.tick()
            step_count += 1
            t_sim = step_count * 0.05
            
            # 車両の物理量を取得
            curr_v = vehicle.get_velocity()
            speed_ms = math.sqrt(curr_v.x**2 + curr_v.y**2 + curr_v.z**2)
            
            # ---------------------------------------------------------
            # 1. Planning (計画) : ウェイポイントと偏差(CTE)の計算
            # ---------------------------------------------------------
            target_wp = planner.get_target_waypoint(lookahead_distance=5.0)
            steering_error = planner.calculate_steering_error(target_wp)
            target_speed = args.target_speed
            
            # Signed Cross-Track Error (CTE) の算出
            vehicle_transform = vehicle.get_transform()
            vehicle_loc = vehicle_transform.location
            carla_map = world.get_map()
            current_wp = carla_map.get_waypoint(vehicle_loc)
            wp_transform = current_wp.transform
            
            vec_wp_to_car = vehicle_loc - wp_transform.location
            right_vec = wp_transform.get_right_vector()
            cte = vec_wp_to_car.x * right_vec.x + vec_wp_to_car.y * right_vec.y + vec_wp_to_car.z * right_vec.z
            
            # ---------------------------------------------------------
            # 2. Control (制御) : 指令値の計算と適用
            # ---------------------------------------------------------
            # センサーデータの取得
            img_rgb = sensor_manager.get_image('rgb_front')
            img_seg = sensor_manager.get_image('seg_front')
            lidar_data = sensor_manager.get_sensor_data('lidar')
            radar_data = sensor_manager.get_sensor_data('radar')
            imu_data = sensor_manager.get_sensor_data('imu')
            gnss_data = sensor_manager.get_sensor_data('gnss')
            
            if lidar_data is not None:
                latest_lidar_points = lidar_data
            if radar_data is not None:
                latest_radar_points = radar_data
                
            # レーダー追尾ターゲット処理
            radar_closest_dist = 999.0
            radar_closest_vel = 0.0
            if latest_radar_points is not None and len(latest_radar_points) > 0:
                closest_idx = np.argmin(latest_radar_points[:, 3])
                radar_closest_dist = latest_radar_points[closest_idx, 3]
                radar_closest_vel = latest_radar_points[closest_idx, 0]
                
            # IMU/GNSS テレメトリのログ
            if imu_data is not None:
                log_imu_t.append(t_sim)
                log_imu_accel.append(imu_data['accel'])
            if gnss_data is not None:
                log_gnss_t.append(t_sim)
                log_gnss_coords.append((gnss_data['lat'], gnss_data['lon']))
                
            # 制御指令値の決定と適用
            if args.use_autopilot:
                control_cmd = vehicle.get_control()
            else:
                control_cmd = controller.run_step(target_speed, steering_error)
                vehicle.apply_control(control_cmd)
                
            # 障害物との距離とAEB自動停止判定
            actual_dist = 999.0
            if obstacle_vehicle and obstacle_vehicle.is_alive:
                obs_loc = obstacle_vehicle.get_transform().location
                actual_dist = vehicle_loc.distance(obs_loc)
                speed_kmh = speed_ms * 3.6
                
                # 自動ブレーキで安全停止したか判定
                if actual_dist < 10.0 and speed_kmh < 0.1:
                    consecutive_stopped += 1
                    if consecutive_stopped >= 15: # 15フレーム(約0.75秒)連続で停止
                        print("\n[SUCCESS] Ego vehicle successfully stopped in front of the obstacle!")
                        print(f"Final distance to obstacle: {actual_dist:.2f} meters.")
                        break
                else:
                    consecutive_stopped = 0
                
                # 衝突判定
                if actual_dist < 3.8:
                    print("\n[COLLISION] Ego vehicle got too close or collided with the obstacle!")
                    break
                    
            if step_count > 500: # 最大25秒でタイムアウト
                print("\n[TIMEOUT] Simulation reached maximum duration.")
                break

            if img_rgb is not None and img_seg is not None:
                # 画面縮小表示（1280x720 からダッシュボード用に480x270にリサイズして見切れ問題を解消）
                img_resized = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_BGRA2BGR), (480, 270))
                seg_resized = cv2.resize(cv2.cvtColor(img_seg, cv2.COLOR_BGRA2BGR), (480, 270))
                
                # 映像の上にタイトル・ラベルを控えめに描画
                cv2.putText(img_resized, "EGO VEHICLE RGB VIEW", (15, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(seg_resized, "SEMANTIC SEGMENTATION VIEW", (15, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                
                # 左右の映像を横に結合 (960x270)
                video_area = np.hstack((img_resized, seg_resized))
                
                # 下部に 150px の HUD (情報表示領域) を結合するため、960x150 の黒背景を作成
                hud_area = np.zeros((150, 960, 3), dtype=np.uint8)
                
                # HUDの列分割境界線を引く (3列構成)
                cv2.line(hud_area, (0, 0), (960, 0), (80, 80, 80), 2)
                cv2.line(hud_area, (310, 0), (310, 150), (80, 80, 80), 1)
                cv2.line(hud_area, (610, 0), (610, 150), (80, 80, 80), 1)
                
                # 結合 (960x420)
                dashboard = np.vstack((video_area, hud_area))
                
                # --- 第1列: Ego Vehicle Control Telemetry ---
                cv2.putText(dashboard, "1. Ego Vehicle Telemetry", (15, 290), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                speed_text = f"Speed: {speed_ms*3.6:.1f} km/h"
                cv2.putText(dashboard, speed_text, (15, 315), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                target_text = f"Target Speed: {target_speed*3.6:.1f} km/h"
                cv2.putText(dashboard, target_text, (15, 340), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                ap_text = f"Autopilot: {'ACTIVE' if args.use_autopilot else 'INACTIVE'}"
                cv2.putText(dashboard, ap_text, (15, 365), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255) if args.use_autopilot else (150, 150, 150), 1, cv2.LINE_AA)
                
                # --- 第2列: Control Commands ---
                cv2.putText(dashboard, "2. Control Commands", (325, 290), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(dashboard, f"Steer Cmd: {control_cmd.steer:.2f}", (325, 315), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(dashboard, f"Throttle: {control_cmd.throttle:.2f} | Brake: {control_cmd.brake:.2f}", (325, 340), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(dashboard, f"CTE (Offset): {cte:.2f} m | Yaw Err: {steering_error:.2f} rad", (325, 365), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # --- 第3列: Multi-Sensor Perception ---
                cv2.putText(dashboard, "3. Multi-Sensor Perception", (625, 290), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                
                radar_info = "No Target"
                if radar_closest_dist < 100.0:
                    radar_info = f"Dist: {radar_closest_dist:.1f}m, RelV: {radar_closest_vel:.1f}m/s"
                cv2.putText(dashboard, f"Radar Closest: {radar_info}", (625, 315), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                lidar_count = len(lidar_data) if lidar_data is not None else 0
                radar_count = len(radar_data) if radar_data is not None else 0
                cv2.putText(dashboard, f"LiDAR: {lidar_count} pts | Radar: {radar_count} dets", (625, 340), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                if imu_data is not None and gnss_data is not None:
                    ax, ay, az = imu_data['accel']
                    cv2.putText(dashboard, f"IMU Accel: [{ax:.1f}, {ay:.1f}, {az:.1f}] | GPS Lat: {gnss_data['lat']:.5f}", (625, 365), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
                    cv2.putText(dashboard, f"Compass: {math.degrees(imu_data['compass']):.1f} deg | GPS Lon: {gnss_data['lon']:.5f}", (625, 385), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
                
                # 障害物警告を中央に表示
                if obstacle_vehicle and obstacle_vehicle.is_alive:
                    if actual_dist < 15.0:
                        cv2.putText(dashboard, f"!!! COLLISION WARNING: {actual_dist:.1f}m !!!", (300, 250), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                
                # 画面表示
                cv2.imshow("CARLA PID Control & Multi-Sensor Dashboard", dashboard)
                cv2.waitKey(1)
                
                # 録画ライターの初期化（初回のみ）
                if not args.no_record and video_writer is None:
                    h, w, _ = dashboard.shape
                    pid_suffix = f"lon_{args.kp_lon}_{args.ki_lon}_{args.kd_lon}_lat_{args.kp_lat}_{args.ki_lat}_{args.kd_lat}"
                    out_path = os.path.join(os.path.dirname(__file__), f"carla_run_PID_{pid_suffix}_recording.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                    print(f"Video recording started: {out_path}")
                
                if video_writer is not None:
                                        video_writer.write(dashboard)
            
            # ---------------------------------------------------------
            # 4. Data Logging (ログ記録)
            # ---------------------------------------------------------
            log_time.append(t_sim)
            log_speed.append(speed_ms)
            log_target_speed.append(target_speed)
            log_steer.append(control_cmd.steer)
            log_throttle.append(control_cmd.throttle)
            log_brake.append(control_cmd.brake)
            log_cte.append(cte)
            log_heading_err.append(steering_error)
            
    except KeyboardInterrupt:
        print("\nSimulation manually stopped by user.")
    finally:
        # --- [ビデオの終了処理] ---
        if video_writer is not None:
            video_writer.release()
            print("Video recording saved.")
            
        # --- [データグラフの生成と保存] ---
        if not args.no_record and len(log_time) > 10:
            print("Generating performance and tuning analysis plots...")
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            # (1) 速度追従
            axs[0, 0].plot(log_time, log_target_speed, 'r--', label='Target Speed')
            axs[0, 0].plot(log_time, log_speed, 'b-', label='Actual Speed')
            axs[0, 0].set_title(f"Speed Control (Kp={args.kp_lon}, Ki={args.ki_lon}, Kd={args.kd_lon})")
            axs[0, 0].set_xlabel("Time (s)")
            axs[0, 0].set_ylabel("Speed (m/s)")
            axs[0, 0].legend()
            axs[0, 0].grid(True)
            
            # (2) クロス・トラック・エラー (CTE)
            axs[0, 1].plot(log_time, log_cte, 'g-', label='Cross-Track Error (m)')
            axs[0, 1].axhline(0.0, color='k', linestyle='--', alpha=0.5)
            axs[0, 1].set_title(f"Lateral Displacement (Kp_lat={args.kp_lat}, Kd_lat={args.kd_lat})")
            axs[0, 1].set_xlabel("Time (s)")
            axs[0, 1].set_ylabel("Error (meters)")
            axs[0, 1].legend()
            axs[0, 1].grid(True)
            
            # (3) ステアリング角操舵指令
            axs[1, 0].plot(log_time, log_steer, 'm-', label='Steering Command')
            axs[1, 0].set_ylim(-1.1, 1.1)
            axs[1, 0].set_title("Steering Command History")
            axs[1, 0].set_xlabel("Time (s)")
            axs[1, 0].set_ylabel("Value [-1.0 (Left) to 1.0 (Right)]")
            axs[1, 0].legend()
            axs[1, 0].grid(True)
            
            # (4) スロットル / ブレーキ指令
            axs[1, 1].plot(log_time, log_throttle, 'c-', label='Throttle')
            axs[1, 1].plot(log_time, log_brake, 'orange', label='Brake')
            axs[1, 1].set_ylim(-0.1, 1.1)
            axs[1, 1].set_title("Pedal Controls History")
            axs[1, 1].set_xlabel("Time (s)")
            axs[1, 1].set_ylabel("Control Input Value [0.0 to 1.0]")
            axs[1, 1].legend()
            axs[1, 1].grid(True)
            
            plt.tight_layout()
            pid_suffix = f"lon_{args.kp_lon}_{args.ki_lon}_{args.kd_lon}_lat_{args.kp_lat}_{args.ki_lat}_{args.kd_lat}"
            plot_path = os.path.join(os.path.dirname(__file__), f"carla_pid_tuning_{pid_suffix}_results.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Tuning results graph saved: {plot_path}")
            
            # (5) センサーデータの可視化プロット生成と保存 (Tutorial Step 7)
            print("Generating multi-sensor analysis plots (LiDAR, Radar, IMU, GNSS)...")
            fig_sensor, axs_sensor = plt.subplots(2, 2, figsize=(16, 12))
            
            # 5.1 LiDAR XY平面 2D散布図 (高さ Z で色分け)
            if latest_lidar_points is not None and len(latest_lidar_points) > 0:
                lx = latest_lidar_points[:, 0]
                ly = latest_lidar_points[:, 1]
                lz = latest_lidar_points[:, 2]
                sc1 = axs_sensor[0, 0].scatter(lx, ly, c=lz, cmap='viridis', s=2, alpha=0.8)
                axs_sensor[0, 0].set_title("LiDAR 2D Top-down View (colored by Height Z)")
                axs_sensor[0, 0].set_xlabel("X (meters)")
                axs_sensor[0, 0].set_ylabel("Y (meters)")
                axs_sensor[0, 0].axis('equal')
                fig_sensor.colorbar(sc1, ax=axs_sensor[0, 0], label='Height Z (meters)')
                axs_sensor[0, 0].grid(True)
            else:
                axs_sensor[0, 0].text(0.5, 0.5, "No LiDAR Data", ha='center', va='center')
                axs_sensor[0, 0].set_title("LiDAR 2D Top-down View")
                
            # 5.2 レーダーデータ (極座標からデカルト座標に変換、相対速度で色分け)
            if latest_radar_points is not None and len(latest_radar_points) > 0:
                # [velocity, azimuth, altitude, depth]
                velocities = latest_radar_points[:, 0]
                azimuths = latest_radar_points[:, 1]
                altitudes = latest_radar_points[:, 2]
                depths = latest_radar_points[:, 3]
                
                # 極座標からデカルト座標 (X, Y, Z) への変換
                rx = depths * np.cos(altitudes) * np.cos(azimuths)
                ry = depths * np.cos(altitudes) * np.sin(azimuths)
                
                sc2 = axs_sensor[0, 1].scatter(rx, ry, c=velocities, cmap='coolwarm', s=30, edgecolor='black', alpha=0.9)
                axs_sensor[0, 1].set_title("Radar Detections (colored by Relative Velocity)")
                axs_sensor[0, 1].set_xlabel("X (meters)")
                axs_sensor[0, 1].set_ylabel("Y (meters)")
                axs_sensor[0, 1].axis('equal')
                fig_sensor.colorbar(sc2, ax=axs_sensor[0, 1], label='Relative Velocity (m/s)')
                axs_sensor[0, 1].grid(True)
            else:
                axs_sensor[0, 1].text(0.5, 0.5, "No Radar Data", ha='center', va='center')
                axs_sensor[0, 1].set_title("Radar Detections")
                
            # 5.3 IMU加速度の時系列折れ線グラフ
            if len(log_imu_accel) > 0:
                imu_accel_arr = np.array(log_imu_accel)
                axs_sensor[1, 0].plot(log_imu_t, imu_accel_arr[:, 0], 'r-', label='Accel X')
                axs_sensor[1, 0].plot(log_imu_t, imu_accel_arr[:, 1], 'g-', label='Accel Y')
                axs_sensor[1, 0].plot(log_imu_t, imu_accel_arr[:, 2], 'b-', label='Accel Z')
                axs_sensor[1, 0].set_title("IMU Accelerometer Time-Series")
                axs_sensor[1, 0].set_xlabel("Time (seconds)")
                axs_sensor[1, 0].set_ylabel("Acceleration (m/s^2)")
                axs_sensor[1, 0].legend()
                axs_sensor[1, 0].grid(True)
            else:
                axs_sensor[1, 0].text(0.5, 0.5, "No IMU Data", ha='center', va='center')
                axs_sensor[1, 0].set_title("IMU Accelerometer Time-Series")
                
            # 5.4 GNSS (GPS) 走行経路の散布図
            if len(log_gnss_coords) > 0:
                gnss_arr = np.array(log_gnss_coords)
                sc4 = axs_sensor[1, 1].scatter(gnss_arr[:, 1], gnss_arr[:, 0], c=log_gnss_t, cmap='plasma', s=15, alpha=0.9)
                axs_sensor[1, 1].set_title("GNSS GPS Route Tracking")
                axs_sensor[1, 1].set_xlabel("Longitude (deg)")
                axs_sensor[1, 1].set_ylabel("Latitude (deg)")
                fig_sensor.colorbar(sc4, ax=axs_sensor[1, 1], label='Time (seconds)')
                axs_sensor[1, 1].grid(True)
            else:
                axs_sensor[1, 1].text(0.5, 0.5, "No GNSS Data", ha='center', va='center')
                axs_sensor[1, 1].set_title("GNSS GPS Route Tracking")
                
            plt.tight_layout()
            pid_suffix = f"lon_{args.kp_lon}_{args.ki_lon}_{args.kd_lon}_lat_{args.kp_lat}_{args.ki_lat}_{args.kd_lat}"
            sensor_plot_path = os.path.join(os.path.dirname(__file__), f"carla_sensor_analysis_{pid_suffix}.png")
            plt.savefig(sensor_plot_path, dpi=150)
            plt.close()
            print(f"Multi-sensor analysis graph saved: {sensor_plot_path}")
            
        # --- [クリーンアップ処理] ---
        print("Restoring CARLA simulator settings & destroying spawned actors...")
        if sensor_manager:
            sensor_manager.destroy()
        if vehicle:
            try:
                vehicle.destroy()
                print("Ego vehicle destroyed.")
            except Exception:
                pass
        if obstacle_vehicle:
            try:
                obstacle_vehicle.destroy()
                print("Obstacle vehicle destroyed.")
            except Exception:
                pass
                
        # 同期モードを解除して終了 (解除しないとCARLAの内部時間がフリーズする)
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
            
        cv2.destroyAllWindows()
        print("Cleanup completed. System shut down.")

if __name__ == '__main__':
    main()
