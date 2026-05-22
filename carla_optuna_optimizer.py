"""
CARLA Real-World Optuna Optimizer
本物のCARLAシミュレータに接続し、同期モードで物理天気パラメータをベイズ最適化（Optuna）を用いて直接操作、
単眼3D物体検出（YOLO3D/v8）のエッジケース（知覚リスクが最小＝AIの見落とし・危険性の過小評価）を高速探索する本番用パイプライン。
"""

import argparse
import carla
import optuna
import numpy as np
import cv2
import time
import os
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
        self.target_vehicle = None
        self.camera = None
        self.last_image = None
        
        # 元の設定をバックアップして同期モードを有効化
        self.original_settings = self.world.get_settings()
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05 # 20 FPS (PID制御のdtと最適に一致)
        self.world.apply_settings(self.settings)
        print("Synchronous mode enabled (dt = 0.05s).")

        # シナリオアクターのセットアップ
        self._setup_scenario()

    def _setup_scenario(self):
        """
        評価用の固定シナリオ（直線接近シナリオ）をロードしてアクターをスポーンします。
        """
        try:
            # 1. 自車 (Ego Vehicle: Tesla Model 3) のスポーン
            ego_bp = self.blueprint_library.filter('model3')[0]
            # Town01などの直線道路上で確実な位置にスポーンするため、最初のスポーンポイントを利用
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points found on the map.")
            
            # 安全な特定の場所を基準にする (ここでは最初のポイント)
            ego_transform = spawn_points[0]
            ego_transform.location.z += 0.5 # 地面へのめり込み防止
            
            self.ego_vehicle = self.world.spawn_actor(ego_bp, ego_transform)
            self.actors.append(self.ego_vehicle)
            print(f"Ego vehicle (Tesla Model 3) spawned at {ego_transform.location}")

            # 2. 相手車 (Target Vehicle: 大型トラック) を30メートル前方に配置
            target_bp = self.blueprint_library.filter('carlacola')[0]
            forward_vector = ego_transform.get_forward_vector()
            target_location = ego_transform.location + forward_vector * 30.0
            target_location.z += 0.5
            target_transform = carla.Transform(target_location, ego_transform.rotation)
            
            self.target_vehicle = self.world.spawn_actor(target_bp, target_transform)
            self.actors.append(self.target_vehicle)
            print(f"Target vehicle (Truck) spawned at {target_location}")

            # 3. フロントガラス上部にRGBカメラを取り付ける
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '1280')
            camera_bp.set_attribute('image_size_y', '720')
            camera_bp.set_attribute('fov', '110')
            
            # フロントガラス上部 (x=2.0, z=1.4) で少し下向き (pitch=-5度)
            camera_transform = carla.Transform(
                carla.Location(x=2.0, y=0.0, z=1.4),
                carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0)
            )
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
            self.actors.append(self.camera)

            # カメラ画像のストリームを同期で待ち受けるコールバックを登録
            def _on_camera_capture(image):
                # CARLA画像バッファからnumpy BGR画像への変換
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                bgr_image = array[:, :, :3]
                self.last_image = bgr_image

            self.camera.listen(_on_camera_capture)
            print("Front Camera mounted and listen callback registered.")

        except Exception as e:
            self.destroy()
            raise e

    def set_weather(self, sun_altitude_angle, precipitation, fog_density):
        """
        CARLAの環境物理ツマミ（天候パラメータ）を動的に操作します。
        """
        weather = carla.WeatherParameters(
            sun_altitude_angle=sun_altitude_angle,
            precipitation=precipitation,
            fog_density=fog_density,
            cloudiness=max(precipitation, fog_density), # 雨や霧に応じて雲量も自動連動
            wetness=precipitation, # 雨量に応じて路面の濡れ（反射）を設定
            wind_intensity=10.0
        )
        self.world.set_weather(weather)
        # 設定を確定するために同期ワールドを進める
        self.world.tick()

    def get_image(self):
        """
        カメラセンサーから最新の画像フレームを取得します。
        """
        # 同期tickを実行してアクターの物理更新とカメラ撮影を進める
        self.world.tick()
        
        # コールバックから画像が届くまで少し待機
        timeout = 2.0
        start_time = time.time()
        while self.last_image is None:
            time.sleep(0.01)
            if time.time() - start_time > timeout:
                print("Warning: Camera frame timeout. Returning dummy image.")
                return np.zeros((720, 1280, 3), dtype=np.uint8)
                
        return self.last_image

    def get_ground_truth(self):
        """
        CARLAシミュレータの内部物理真値 (Ground Truth) をダイレクトに取得します。
        """
        ego_loc = self.ego_vehicle.get_transform().location
        ego_vel = self.ego_vehicle.get_velocity()
        
        target_loc = self.target_vehicle.get_transform().location
        target_vel = self.target_vehicle.get_velocity()
        
        return {
            'ego_pos': [ego_loc.x, ego_loc.y, ego_loc.z],
            'ego_vel': [ego_vel.x, ego_vel.y, ego_vel.z],
            'target_pos': [target_loc.x, target_loc.y, target_loc.z],
            'target_vel': [target_vel.x, target_vel.y, target_vel.z],
            'target_class': 'truck'
        }

    def reset_actors_physics(self):
        """
        アクターの位置と速度を初期の安全な状態に物理リセットします。
        """
        # 自車の再配置 (初期位置へ)
        spawn_points = self.world.get_map().get_spawn_points()
        ego_transform = spawn_points[0]
        ego_transform.location.z += 0.5
        self.ego_vehicle.set_transform(ego_transform)
        self.ego_vehicle.set_target_velocity(carla.Vector3D(x=15.0, y=0.0, z=0.0)) # 15 m/s (~54 km/h)で前進
        
        # 相手車の再配置 (自車の30m前方で停止状態に固定)
        forward_vector = ego_transform.get_forward_vector()
        target_location = ego_transform.location + forward_vector * 30.0
        target_location.z += 0.5
        target_transform = carla.Transform(target_location, ego_transform.rotation)
        self.target_vehicle.set_transform(target_transform)
        self.target_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0)) # 停止車

        # 物理エンジンを落ち着かせるために同期モードで数フレームTick
        for _ in range(5):
            self.world.tick()
        self.last_image = None

    def destroy(self):
        """
        アクターをすべて破棄し、CARLAの設定を元に戻すクリーンアップ処理（最重要）。
        """
        print("Cleaning up CARLA real actors and restoring settings...")
        for actor in self.actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actors = []
        
        # 同期モードを解除して元のシミュレータ設定を復元
        try:
            self.world.apply_settings(self.original_settings)
            print("CARLA original settings restored successfully.")
        except Exception as e:
            print(f"Error restoring CARLA settings: {e}")


def run_real_carla_optimization(n_trials=30, sampler_name='TPE'):
    """
    ライブCARLAシミュレータとOptunaを接続し、最適化探索ループを実行します。
    """
    os.makedirs("results", exist_ok=True)
    
    # ライブ環境および評価モジュールのインスタンス化
    env = RealCarlaEnv()
    evaluator = YoloEvaluator()
    risk_calculator = RiskCalculator()
    
    # 探索に使用するサンプラー
    if sampler_name == 'Random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
        
    study = optuna.create_study(direction="minimize", sampler=sampler)
    history = []
    
    best_score_so_far = float('inf')
    
    try:
        def objective(trial):
            nonlocal best_score_so_far
            start_time = time.time()
            
            # 各試行の前に車両の位置・速度をリセット（初期配置を固定）
            env.reset_actors_physics()
            
            # Optunaによる天候パラメータ（ツマミ）の決定
            sun_altitude_angle = trial.suggest_float("sun_altitude_angle", -15.0, 90.0) # 夜から昼、特に夕方西日を重点探索
            precipitation = trial.suggest_float("precipitation", 0.0, 100.0) # 雨量
            fog_density = trial.suggest_float("fog_density", 0.0, 100.0) # 霧の濃さ
            
            # CARLAに天候を設定
            env.set_weather(sun_altitude_angle, precipitation, fog_density)
            
            # 霧や雨による光学的な見え方を安定させるため、また接近による「手遅れシナリオ」をエミュレートするため
            # 同期モードで15フレーム（0.75秒間）進行させ、接近しながら認識テストを行う
            for _ in range(15):
                img = env.get_image()
            
            # 最も接近した状態（危険度のピーク）でのカメラ画像を取得して認識
            img = env.get_image()
            
            # YOLO（YOLO3Dエミュレータ）による認識（estimated Z-distance）
            min_z_dist, confidence, annotated_img = evaluator.evaluate(img, return_image=True)
            
            # CARLAの物理真値（GT）を取得
            gt = env.get_ground_truth()
            
            # 知覚リスクの算出
            # ※YOLOが雨や霧で見落とす（Z距離=無限大）と、知覚リスク R_perceived は 0 に近づきます。
            # ※本研究では「本当は危険（GTが至近距離）なのに、AIが完全に安全だと誤認（R=0）している極限の脆弱性」を探索するため、
            #   Optunaは R_perceived が『最小』になるパラメータを探索（minimize）します。
            r_perceived, debug_info = risk_calculator.calculate_risk(
                ego_pos=gt['ego_pos'], ego_vel=gt['ego_vel'],
                target_pos=gt['target_pos'], target_vel=gt['target_vel'],
                target_class=gt['target_class'],
                yolo_z_distance=min_z_dist
            )
            
            score = r_perceived
            
            # エッジケース（最も過小評価した最悪のケース）が更新された場合、結果画像を保存
            if score < best_score_so_far:
                best_score_so_far = score
                cv2.imwrite(f"results/real_edge_case_worst_{sampler_name}.jpg", annotated_img)
                print(f"--> [NEW WORST EDGE CASE FOUND] Perceived Risk Score (Minimized): {score:.4f}")
                print(f"    Params: Sun={sun_altitude_angle:.2f}, Rain={precipitation:.2f}, Fog={fog_density:.2f}")
                print(f"    YOLO Estimated Z: {min_z_dist:.2f}m (GT Distance: {debug_info['gt_distance']:.2f}m)")
                
            elapsed_time = time.time() - start_time
            
            # 実験履歴の記録
            history.append({
                "trial": trial.number,
                "sun_altitude_angle": sun_altitude_angle,
                "precipitation": precipitation,
                "fog_density": fog_density,
                "yolo_z_distance": min_z_dist,
                "gt_distance": debug_info['gt_distance'],
                "r_perceived": r_perceived,
                "omega": debug_info['omega'],
                "alpha": debug_info['alpha'],
                "beta": debug_info['beta'],
                "elapsed_time_sec": elapsed_time
            })
            
            return score

        # 最適化探索の実行
        study.optimize(objective, n_trials=n_trials)
        
    finally:
        # 必ずアクターを破棄してCARLAを元の設定に戻す（クラッシュ防止）
        env.destroy()

    # 履歴をCSVに保存
    df_history = pd.DataFrame(history)
    df_history.to_csv(f"results/real_history_{sampler_name}.csv", index=False)
    
    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA + Optuna edge-case search")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials (default: 30)")
    parser.add_argument("--sampler", type=str, default="TPE", choices=["TPE", "Random"],
                        help="Sampler: TPE or Random (default: TPE)")
    args = parser.parse_args()

    print("====================================================")
    print("  CARLA Live Optuna Edge-Case Discovery Pipeline")
    print(f"  Trials: {args.trials}, Sampler: {args.sampler}")
    print("====================================================")
    
    try:
        study = run_real_carla_optimization(n_trials=args.trials, sampler_name=args.sampler)
        print(f"\n{args.sampler} Optimization complete!")
        print(f"Worst Edge Case (Illusion of Safety) Perceived Risk: {study.best_trial.value:.4f}")
        print(f"Weather parameters: {study.best_trial.params}")
        
    except Exception as e:
        print(f"\nFailed to execute pipeline: {e}")
        print("Please ensure CARLA Simulator is running (e.g. Town01 map loaded) and the python-carla library is installed.")
