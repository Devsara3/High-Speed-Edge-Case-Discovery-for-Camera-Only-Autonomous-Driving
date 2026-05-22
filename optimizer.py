import optuna
import pandas as pd
import time
import os
import cv2
from carla_mock import MockCarlaEnv
from evaluator import YoloEvaluator
from risk_calculator import RiskCalculator

def run_optimization(env, evaluator, risk_calculator, n_trials=50, sampler_name='TPE'):
    """
    Optunaを用いてエッジケース（認識率が最も下がる天候パラメータ）を探索します。
    """
    print(f"Starting optimization with {sampler_name} sampler for {n_trials} trials...")
    
    # サンプラーの選択
    if sampler_name == 'Random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:
        # TPE (Tree-structured Parzen Estimator): デフォルトの効率的な探索アルゴリズム
        sampler = optuna.samplers.TPESampler(seed=42)
 
    study = optuna.create_study(direction="maximize", sampler=sampler)
 
    # トライアルごとの実行時間と結果を記録するリスト
    history = []
    
    # 最悪のエッジケース（乖離スコアが最大）の追跡
    worst_edge_case_score = -float('inf')
    # 最良のケース（乖離スコアが最小＝正しく認識）の追跡
    best_case_score = float('inf')
 
    def objective(trial):
        nonlocal worst_edge_case_score, best_case_score
        start_time = time.time()
        
        # 探索空間の定義
        # -15度〜90度（夜〜真昼）
        sun_altitude_angle = trial.suggest_float("sun_altitude_angle", -15.0, 90.0)
        # 0〜100%（晴れ〜大雨）
        precipitation = trial.suggest_float("precipitation", 0.0, 100.0)
        # 0〜100%（霧なし〜濃霧）
        fog_density = trial.suggest_float("fog_density", 0.0, 100.0)
 
        # 環境の更新と画像取得
        env.set_weather(sun_altitude_angle, precipitation, fog_density)
        img = env.get_image()
 
        # 評価（YOLO3Dによる主観的Z距離の取得）
        min_z_distance, conf, annotated_img = evaluator.evaluate(img, return_image=True)
        
        # モック環境からGround Truthを取得
        gt = env.get_ground_truth()
        
        # 知覚リスクの計算
        r_perceived, debug_info = risk_calculator.calculate_risk(
            ego_pos=gt['ego_pos'], ego_vel=gt['ego_vel'],
            target_pos=gt['target_pos'], target_vel=gt['target_vel'],
            target_class=gt['target_class'],
            yolo_z_distance=min_z_distance
        )
        
        # 真値（GT）に基づいた物理リスクの計算
        r_gt, debug_info_gt = risk_calculator.calculate_risk(
            ego_pos=gt['ego_pos'], ego_vel=gt['ego_vel'],
            target_pos=gt['target_pos'], target_vel=gt['target_vel'],
            target_class=gt['target_class'],
            yolo_z_distance=debug_info['gt_distance']
        )
        
        # 知覚ギャップスコア
        score = r_gt - r_perceived
        
        # エッジケース（スコアが最大＝ギャップが最大）が更新された場合
        if score > worst_edge_case_score:
            worst_edge_case_score = score
            cv2.imwrite(f"results/edge_case_worst_{sampler_name}.jpg", annotated_img)
            
        # 逆に最もよく認識できた状態（スコアが最小＝ギャップが最小）を記録
        if score < best_case_score:
            best_case_score = score
            cv2.imwrite(f"results/edge_case_best_{sampler_name}.jpg", annotated_img)
 
        elapsed_time = time.time() - start_time
        
        # 履歴に記録
        history.append({
            "trial": trial.number,
            "sun_altitude_angle": sun_altitude_angle,
            "precipitation": precipitation,
            "fog_density": fog_density,
            "min_z_distance": min_z_distance,
            "gt_distance": debug_info['gt_distance'],
            "r_gt": r_gt,
            "r_perceived": r_perceived,
            "perception_gap": score,
            "omega": debug_info['omega'],
            "alpha": debug_info['alpha'],
            "beta": debug_info['beta'],
            "elapsed_time_sec": elapsed_time
        })
 
        return score
 
    # 最適化の実行
    study.optimize(objective, n_trials=n_trials)
    
    print("\nBest Trial:")
    print(f"  Score (Maximized Perception Gap): {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
 
    return study, pd.DataFrame(history)
 
if __name__ == "__main__":
    # モック環境とエバリュエータの初期化
    base_image_path = "base_image.png"
    env = MockCarlaEnv(base_image_path)
    evaluator = YoloEvaluator()
    risk_calc = RiskCalculator()
    
    os.makedirs("results", exist_ok=True)
 
    # 1. TPE (提案手法) による探索 (50回)
    study_tpe, df_tpe = run_optimization(env, evaluator, risk_calc, n_trials=50, sampler_name='TPE')
    df_tpe.to_csv("results/history_tpe.csv", index=False)
    
    # 2. ランダム探索 (ベースライン) との比較 (50回)
    study_random, df_random = run_optimization(env, evaluator, risk_calc, n_trials=50, sampler_name='Random')
    df_random.to_csv("results/history_random.csv", index=False)
    
    print("\nOptimization completed. Results saved in 'results' directory.")
