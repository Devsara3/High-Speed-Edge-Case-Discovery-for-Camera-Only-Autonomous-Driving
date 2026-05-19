# -*- coding: utf-8 -*-
"""
CARLA Integration Demo (統合メインループ & 制御チューニング用スクリプト)

このスクリプトは、Gitリポジトリをクローンした開発者が、
自身のPC環境でCARLAサーバーに接続し、純粋なPID制御（速度・ステアリング）の
挙動確認およびパラメータ調整（チューニング）を行うためのプログラムです。
※ AI関連の重いライブラリ（PyTorch等）の依存関係は一切ありません。
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
    parser = argparse.ArgumentParser(description="CARLA Pure PID Control & Tuning Script")
    
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
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("====================================================")
    print(" CARLA PID Autonomous Driving Control System")
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
        
    # シミュレータ同期モードの設定（シミュレータとPython制御周期を完全に一致させる）
    settings = world.get_settings()
    original_settings = world.get_settings()  # 終了時の復元用
    
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS (PIDのdtと一致させる)
    world.apply_settings(settings)
    
    vehicle = None
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
    
    try:
        # --- [車両とセンサーのスポーン] ---
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]  # Tesla Model 3
        spawn_points = world.get_map().get_spawn_points()
        
        # 最初のスポーンポイントを選択
        spawn_point = spawn_points[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print("Vehicle spawned successfully.")
        
        # モジュール初期化
        sensor_manager = CarlaSensorManager(world, vehicle)
        planner = CarlaWaypointPlanner(world, vehicle)
        
        # コントローラー初期化（引数からゲインを設定）
        controller = CarlaPIDController(vehicle, dt=0.05)
        controller.lon_kp, controller.lon_ki, controller.lon_kd = args.kp_lon, args.ki_lon, args.kd_lon
        controller.lat_kp, controller.lat_ki, controller.lat_kd = args.kp_lat, args.ki_lat, args.kd_lat
        
        # フロントカメラ（RGB）の設置
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # ルーフ前部
        sensor_manager.spawn_rgb_camera(camera_transform, role_name='rgb_front')
        
        print("Starting Autonomous Loop... (Press Ctrl+C to stop)")
        
        # カメラセンサーが画像を流し始めるまで数フレーム同期クロックを進める
        for _ in range(10):
            world.tick()
            
        start_time = time.time()
        step_count = 0
        
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
            
            # 評価用の正確なSigned Cross-Track Error (CTE) の算出
            vehicle_transform = vehicle.get_transform()
            vehicle_loc = vehicle_transform.location
            carla_map = world.get_map()
            current_wp = carla_map.get_waypoint(vehicle_loc)
            wp_transform = current_wp.transform
            
            # 最寄り車線中心からのズレベクトルと右ベクトルの内積で符号付き横ズレ量(CTE)を算出
            vec_wp_to_car = vehicle_loc - wp_transform.location
            right_vec = wp_transform.get_right_vector()
            cte = vec_wp_to_car.x * right_vec.x + vec_wp_to_car.y * right_vec.y + vec_wp_to_car.z * right_vec.z
            
            # ---------------------------------------------------------
            # 2. Control (制御) : PID指令値の計算と適用
            # ---------------------------------------------------------
            control_cmd = controller.run_step(target_speed=target_speed, target_steering_angle=steering_error)
            vehicle.apply_control(control_cmd)
            
            # ---------------------------------------------------------
            # 3. Visualization (描画) : カメラ画像とテレメトリの表示
            # ---------------------------------------------------------
            img_rgb = sensor_manager.get_image('rgb_front')
            if img_rgb is not None:
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                # 画面上にリアルタイムのテレメトリ情報をオーバーレイ
                cv2.putText(img_bgr, f"Speed: {speed_ms*3.6:.1f} km/h (Target: {target_speed*3.6:.1f})", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img_bgr, f"Cross-Track Error (CTE): {cte:.2f} m", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(img_bgr, f"Steer Cmd: {control_cmd.steer:.2f}", (20, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img_bgr, f"Throttle: {control_cmd.throttle:.2f} | Brake: {control_cmd.brake:.2f}", (20, 160), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("CARLA PID Tracking Control", img_bgr)
                cv2.waitKey(1)
                
                # 録画ライターの初期化（初回のみ）
                if not args.no_record and video_writer is None:
                    h, w, _ = img_bgr.shape
                    out_path = os.path.join(os.path.dirname(__file__), "carla_run_recording.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                    print(f"Video recording started: {out_path}")
                
                if video_writer is not None:
                    video_writer.write(img_bgr)
            
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
            plot_path = os.path.join(os.path.dirname(__file__), "carla_pid_tuning_results.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Tuning results graph saved: {plot_path}")
            
        # --- [クリーンアップ処理] ---
        print("Restoring CARLA simulator settings & destroying spawned actors...")
        if sensor_manager:
            sensor_manager.destroy()
        if vehicle:
            try:
                vehicle.destroy()
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
