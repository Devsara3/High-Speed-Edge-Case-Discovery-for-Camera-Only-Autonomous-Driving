import optuna
import numpy as np
import pandas as pd
import time
import os
import cv2
from carla_mock import MockCarlaEnv
from evaluator import YoloEvaluator
from risk_calculator import RiskCalculator

def run_optimization(env, evaluator, risk_calculator, n_trials=5, sampler_name='TPE', traffic_light_color='red'):
    """
    Optunaを用いてエッジケース（認識率が最も下がる天候パラメータ）を探索します。
    標準RGBカメラのみを評価し、走行時系列全体の最大知覚ギャップ（脆弱性）を最大化します。
    """
    print(f"\n[INFO] Starting optimization with {sampler_name} sampler for {n_trials} trials...")
    print(f"[INFO] Traffic Light Color set to: {traffic_light_color}")
    
    # サンプラーの選択
    if sampler_name == 'Random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
 
    study = optuna.create_study(direction="maximize", sampler=sampler)
 
    history = []
    worst_gap = -float('inf')
 
    def objective(trial):
        nonlocal worst_gap
        start_time = time.time()
        
        # 探索空間の定義
        sun_altitude_angle = trial.suggest_float("sun_altitude_angle", -15.0, 90.0)
        precipitation = trial.suggest_float("precipitation", 0.0, 100.0)
        fog_density = trial.suggest_float("fog_density", 0.0, 100.0)
 
        # 環境の初期化と天候・信号の設定
        env.reset()
        env.set_weather(sun_altitude_angle, precipitation, fog_density)
        env.set_traffic_light_color(traffic_light_color)
        
        max_run_gap = -float('inf')
        worst_run_info = {
            'worst_obstacle': 'unknown',
            'worst_yolo_distance': float('inf'),
            'worst_gt_distance': 0.0,
            'r_gt': 0.0,
            'r_perceived': 0.0
        }
        collision_detected = False
        max_steps = 60 # 最大走行ステップ数 (20 FPS なので約3秒)
        
        for step in range(max_steps):
            # 1. カメラ画像取得と YOLO3D 認識
            img = env.get_image()
            detections, annotated = evaluator.evaluate_multi(img, return_image=True)
            gt = env.get_ground_truth()
            
            # 2. YOLO3D 認識結果に基づくAEB（自動緊急ブレーキ）の制御決定
            apply_brake = False
            for det in detections:
                cls = det['class']
                z = det['z_distance']
                
                # 物理的な障害物 (歩行者、先行車、工事コーン) が15m以内
                if cls in ['pedestrian', 'car', 'construction_signal']:
                    if z <= 15.0:
                        apply_brake = True
                        break
                # 赤/黄信号が18m以内
                elif cls == 'traffic_light':
                    color = det.get('traffic_light_color', 'unknown')
                    if color in ['red', 'yellow', 'unknown'] and z <= 18.0:
                        apply_brake = True
                        break
            
            # 行動決定: 加減速 [throttle_or_brake, steer]
            action = [-1.0, 0.0] if apply_brake else [0.3, 0.0]
            
            # 3. シミュレーションを 1 ステップ進行
            next_img, next_gt = env.step(action)
            
            # 4. このステップでの知覚リスクおよび知覚ギャップの計算
            r_perc, r_gt, gap, info = risk_calculator.calculate_multi_risk(
                ego_pos=gt['ego_pos'], ego_vel=gt['ego_vel'],
                gt_obstacles=gt['obstacles'], yolo_detections=detections
            )
            
            # 走行時系列全体の最悪危険度 (ギャップ最大) の瞬間を追跡
            if gap > max_run_gap:
                max_run_gap = gap
                worst_run_info = {
                    'worst_obstacle': info['worst_obstacle'],
                    'worst_yolo_distance': info['worst_yolo_distance'],
                    'worst_gt_distance': info['worst_gt_distance'],
                    'r_gt': r_gt,
                    'r_perceived': r_perc,
                    'annotated_image': annotated
                }
                
            # 簡易衝突判定 (信号機以外のアクターとの水平距離が 1.5m 未満)
            for obs in gt['obstacles']:
                if obs['class'] != 'traffic_light':
                    dist = np.linalg.norm(np.array(obs['pos'][:2]) - np.array(gt['ego_pos'][:2]))
                    if dist < 1.5:
                        collision_detected = True
                        break
            
            if collision_detected:
                break
                
            # ゴールライン到達判定 (Ego車が 50m 地点に到達)
            if gt['ego_pos'][0] >= 50.0:
                break
 
        # トライアル全体で最悪の天候画像（ギャップが最も開いた危険な瞬間）を保存
        if max_run_gap > worst_gap:
            worst_gap = max_run_gap
            if 'annotated_image' in worst_run_info:
                cv2.imwrite(f"results/edge_case_worst_{sampler_name}_{traffic_light_color}.jpg", worst_run_info['annotated_image'])
                print(f"--> [NEW WORST DYNAMIC EDGE CASE] Gap Score: {max_run_gap:.4f} (Obstacle: {worst_run_info['worst_obstacle']})")
                print(f"    Params: Sun={sun_altitude_angle:.2f}, Rain={precipitation:.2f}, Fog={fog_density:.2f}")
 
        elapsed_time = time.time() - start_time
        
        # 履歴に記録
        history.append({
            "trial": trial.number,
            "sun_altitude_angle": sun_altitude_angle,
            "precipitation": precipitation,
            "fog_density": fog_density,
            "traffic_light_color": traffic_light_color,
            "yolo_z_distance": worst_run_info['worst_yolo_distance'],
            "gt_distance": worst_run_info['worst_gt_distance'],
            "r_gt": worst_run_info['r_gt'],
            "r_perceived": worst_run_info['r_perceived'],
            "gap": max_run_gap,
            "worst_obstacle": worst_run_info['worst_obstacle'],
            "collision": collision_detected,
            "elapsed_time_sec": elapsed_time
        })
 
        return max_run_gap
 
    # 最適化の実行
    study.optimize(objective, n_trials=n_trials)
    
    print("\nBest Trial (RGB Worst Edge Case):")
    print(f"  Max Perception Gap Score: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
 
    return study, pd.DataFrame(history)
 
if __name__ == "__main__":
    # モック環境と評価器の初期化
    base_image_path = "base_image.png"
    env = MockCarlaEnv(base_image_path)
    evaluator = YoloEvaluator()
    risk_calc = RiskCalculator()
    
    os.makedirs("results", exist_ok=True)
 
    # 異なる信号の色 (赤、青/緑) で脆弱性分析を実行
    colors = ['red', 'green']
    
    for color in colors:
        print(f"\n==================== Running Optimization for Traffic Light: {color.upper()} ====================")
        
        # 1. TPE サンプラーでの探索 (5回)
        study_tpe, df_tpe = run_optimization(env, evaluator, risk_calc, n_trials=5, sampler_name='TPE', traffic_light_color=color)
        df_tpe.to_csv(f"results/history_tpe_{color}.csv", index=False)
        
        # 2. Random サンプラーでの探索 (5回)
        study_random, df_random = run_optimization(env, evaluator, risk_calc, n_trials=5, sampler_name='Random', traffic_light_color=color)
        df_random.to_csv(f"results/history_random_{color}.csv", index=False)
    
    print("\n[SUCCESS] Optimization runs completed. Data files saved in 'results/'")
