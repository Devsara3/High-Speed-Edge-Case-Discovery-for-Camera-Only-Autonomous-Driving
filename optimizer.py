import optuna
import pandas as pd
import time
import os
import cv2
from carla_mock import MockCarlaEnv
from evaluator import YoloEvaluator
from risk_calculator import RiskCalculator

def run_optimization(env, evaluator, risk_calculator, n_trials=50, sampler_name='TPE', traffic_light_color='red'):
    """
    Optunaを用いてエッジケース（認識率が最も下がる天候パラメータ）を探索します。
    標準RGBカメラのみを評価し、知覚ギャップ（脆弱性）を最大化します。
    """
    print(f"\n[INFO] Starting optimization with {sampler_name} sampler for {n_trials} trials...")
    print(f"[INFO] Traffic Light Color set to: {traffic_light_color}")
    
    # 信号の色を設定
    env.set_traffic_light_color(traffic_light_color)
    
    # サンプラーの選択
    if sampler_name == 'Random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        sampler = optuna.samplers.TPESampler(seed=42)
 
    study = optuna.create_study(direction="maximize", sampler=sampler)
 
    history = []
    
    # 最悪ケース（ギャップ最大）のスコアと画像を管理
    worst_gap = -float('inf')
 
    def objective(trial):
        nonlocal worst_gap
        start_time = time.time()
        
        # 探索空間の定義
        # -15度〜90度（夜〜真昼）
        sun_altitude_angle = trial.suggest_float("sun_altitude_angle", -15.0, 90.0)
        # 0〜100%（晴れ〜大雨）
        precipitation = trial.suggest_float("precipitation", 0.0, 100.0)
        # 0〜100%（霧なし〜濃霧）
        fog_density = trial.suggest_float("fog_density", 0.0, 100.0)
 
        # 環境の天候更新
        env.set_weather(sun_altitude_angle, precipitation, fog_density)
        
        # ------------------ RGBカメラの評価 ------------------
        img = env.get_image()
        detections, annotated = evaluator.evaluate_multi(img, return_image=True)
        
        # 地面情報（GT）を取得
        gt = env.get_ground_truth()
        
        r_perc, r_gt, gap, info = risk_calculator.calculate_multi_risk(
            ego_pos=gt['ego_pos'], ego_vel=gt['ego_vel'],
            gt_obstacles=gt['obstacles'], yolo_detections=detections
        )
        
        # 最悪ケースの更新と画像保存
        if gap > worst_gap:
            worst_gap = gap
            cv2.imwrite(f"results/edge_case_worst_{sampler_name}_{traffic_light_color}.jpg", annotated)
 
        elapsed_time = time.time() - start_time
        
        # 履歴に記録
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
 
        # 最適化の指標としては「知覚ギャップ」を最大化することを目指す
        # (最も認識が破壊される過酷な環境を探索)
        return gap
 
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
        
        # 1. TPE サンプラーでの探索 (30回)
        study_tpe, df_tpe = run_optimization(env, evaluator, risk_calc, n_trials=30, sampler_name='TPE', traffic_light_color=color)
        df_tpe.to_csv(f"results/history_tpe_{color}.csv", index=False)
        
        # 2. Random サンプラーでの探索 (30回)
        study_random, df_random = run_optimization(env, evaluator, risk_calc, n_trials=30, sampler_name='Random', traffic_light_color=color)
        df_random.to_csv(f"results/history_random_{color}.csv", index=False)
    
    print("\n[SUCCESS] Optimization runs completed. Data files saved in 'results/'")
