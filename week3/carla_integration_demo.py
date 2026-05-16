"""
CARLA Integration Demo (統合メインループ)
将来の拡張性を考慮し、モジュール化された各コンポーネント（Perception / Planning / Control）
を結合し、自動運転とAI推論のループを同期モードで回すメインスクリプトです。
"""
import time
import sys
import numpy as np

# ====== 作成したモジュール群をインポート ======
# 1. 制御モジュール
from carla_pid_controller import CarlaPIDController
# 2. CARLA連携モジュール（今回作成）
from carla_sensor_manager import CarlaSensorManager
from carla_waypoint_planner import CarlaWaypointPlanner
# 3. AI認識モジュール
from depth_estimation import MiDaS_DepthEstimator
# from lane_segmentation import DeepLabV3LaneDetector # 車線認識を使う場合はこちら

try:
    import carla
except ImportError:
    print("CARLAモジュールが見つかりません。CARLAのPython環境で実行してください。")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("OpenCVが見つかりません。pip install opencv-python を実行してください。")
    sys.exit(1)

def main():
    print("Connecting to CARLA Server...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # --- [AIモデルの事前ロード] ---
    print("Loading AI Model (MiDaS Depth Estimation)...")
    midas = MiDaS_DepthEstimator(model_type="MiDaS_small")
    
    # 同期モードの設定（シミュレータの進行時間をPythonスクリプトの処理とピッタリ合わせる必須設定）
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05 # 20FPS (PIDのdt=0.05と一致させる)
    world.apply_settings(settings)
    
    vehicle = None
    sensor_manager = None
    
    try:
        # --- [環境のセットアップ] ---
        # 1. 車両のスポーン
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0] # Tesla Model 3
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        print("Vehicle spawned successfully.")

        # 2. モジュールの初期化
        sensor_manager = CarlaSensorManager(world, vehicle)
        planner = CarlaWaypointPlanner(world, vehicle)
        controller = CarlaPIDController(vehicle, dt=0.05)
        
        # 3. カメラ（センサー）の設置
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4)) # 車のルーフ付近
        sensor_manager.spawn_rgb_camera(camera_transform, role_name='rgb_front')
        
        print("Starting Autonomous Loop... (Press Ctrl+C to stop)")
        # カメラ画像がシミュレータから流れてくるまで数フレーム空回しして待つ
        for _ in range(10):
            world.tick()

        # ====== メインループ ======
        while True:
            # シミュレーションの時間を1コマ（0.05秒）進める
            world.tick()
            
            # ---------------------------------------------------------
            # 1. Perception (認識) : カメラ画像の取得とAIによる推論
            # ---------------------------------------------------------
            img_rgb = sensor_manager.get_image('rgb_front')
            if img_rgb is not None:
                # AI（MiDaS）による深度マップの計算
                depth_map = midas.estimate(img_rgb)
                
                # 結果を可視化するためのOpenCV処理（色付け）
                depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # OpenCV表示用にBGRに戻す
                
                # 元画像とAI結果を横に並べて画面に表示
                display_img = np.hstack((img_bgr, depth_vis))
                cv2.imshow("CARLA Integration (Perception)", display_img)
                cv2.waitKey(1)

            # ---------------------------------------------------------
            # 2. Planning (計画) : パスプランナーによるルート計算
            # ---------------------------------------------------------
            # 少し先（5m先）のルート座標を取得し、現在の車の向きとのエラー（ズレ）を計算
            target_wp = planner.get_target_waypoint(lookahead_distance=5.0)
            steering_error = planner.calculate_steering_error(target_wp)
            target_speed = planner.get_target_speed()
            
            # ---------------------------------------------------------
            # 3. Control (制御) : PIDによるペダル・ハンドル操作
            # ---------------------------------------------------------
            # プランナーが計算した「目標値」と「エラー」をPIDに渡し、車の操作コマンドを取得
            control_cmd = controller.run_step(target_speed=target_speed, target_steering_angle=steering_error)
            
            # 車両に操作コマンドを適用して実際に動かす
            vehicle.apply_control(control_cmd)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        # --- [クリーンアップ] ---
        # 終了時に車やカメラがシミュレータ上に残らないようにお掃除する
        print("Destroying actors and restoring CARLA settings...")
        if sensor_manager:
            sensor_manager.destroy()
        if vehicle:
            vehicle.destroy()
            
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
