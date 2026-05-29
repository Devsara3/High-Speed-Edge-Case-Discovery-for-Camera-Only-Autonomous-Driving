"""
CARLA / Mock ADAS Optuna Optimizer
Optunaを用いて、カメラ単一（Camera-Only）走行時におけるカメラ認識障害を誘発する
最悪の天候物理パラメータ（エッジケース）を自動探索します。
"""

import argparse
import os
import time
import pandas as pd
import optuna
from run_camera_only_experiment import CameraOnlyExperiment

def run_optuna_search(n_trials=5, sampler_name='TPE', scenario_name='sequence', demo_mode=True):
    """
    Optunaによる天候パラメータ探索を実行。
    """
    print("====================================================")
    print("  CARLA/Mock Camera-Only ADAS Weather Optuna Search")
    print(f"  Trials: {n_trials}, Sampler: {sampler_name}, Scenario: {scenario_name}")
    print(f"  Execution Mode: {'MOCK DEMO' if demo_mode else 'REAL CARLA'}")
    print("====================================================")
    
    os.makedirs("results", exist_ok=True)
    
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
        
        # 探索空間の設定
        sun_altitude_angle = trial.suggest_float("sun_altitude_angle", -15.0, 90.0)
        precipitation = trial.suggest_float("precipitation", 0.0, 100.0)
        fog_density = trial.suggest_float("fog_density", 0.0, 100.0)
        
        # 実験インスタンスの初期化
        experiment = CameraOnlyExperiment(demo_mode=demo_mode)
        
        edge_case_img_file = ""
        try:
            # 天候適用
            experiment.set_weather_params(sun_altitude_angle, precipitation, fog_density)
            
            # シナリオの実行
            if scenario_name == 'sequence':
                experiment.run_sequence()
            else:
                experiment.run_experiment(scenario_name)
                
            # 最大認識ギャップと安全指標を取得
            max_gap = experiment.get_max_gap()
            min_dist = experiment.get_min_distance()
            collisions = experiment.get_collision_count()
            worst_params = experiment.get_worst_case_parameters()
            
            # 最悪結果の更新と可視化グラフ保存
            if max_gap > worst_gap:
                worst_gap = max_gap
                experiment.visualize_and_save(f"results/optuna_worst_{scenario_name}.png")
                
                worst_img = experiment.get_worst_image()
                if worst_img is not None:
                    import cv2
                    img_filename = f"edge_case_trial_{trial.number}.png"
                    cv2.imwrite(os.path.join("results", img_filename), worst_img)
                    edge_case_img_file = img_filename
                    
                print(f"\n--> [NEW WORST WEATHER DISCOVERD] Gap: {max_gap:.4f} | MinDist: {min_dist:.2f}m | Collisions: {collisions}")
                print(f"    Params: Sun Alt={sun_altitude_angle:.2f}, Rain={precipitation:.2f}, Fog={fog_density:.2f}\n")
                
        finally:
            experiment.shutdown()
            
        elapsed_time = time.time() - start_time
        
        trial_dict = {
            "trial": trial.number,
            "sun_altitude_angle": sun_altitude_angle,
            "precipitation": precipitation,
            "fog_density": fog_density,
            "gap": max_gap,
            "min_distance": min_dist,
            "collisions": collisions,
            "edge_case_image": edge_case_img_file,
            "elapsed_time_sec": elapsed_time
        }
        trial_dict.update(worst_params)
        history.append(trial_dict)
        
        return max_gap

    study.optimize(objective, n_trials=n_trials)
    
    # 歴史データをCSVに記録
    df_history = pd.DataFrame(history)
    df_history.to_csv(f"results/optuna_history_{scenario_name}_{sampler_name}.csv", index=False)
    
    # Vulnerability Map (Scatter plot of Weather vs Gap)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(df_history['fog_density'], df_history['precipitation'], c=df_history['gap'], cmap='jet', s=100, alpha=0.8)
    plt.colorbar(sc, label='Perception Gap')
    plt.xlabel('Fog Density')
    plt.ylabel('Precipitation')
    plt.title('Weather Vulnerability Map (Gap vs Fog/Rain)')
    plt.grid(True)
    plt.savefig(f"results/vulnerability_map_{scenario_name}.png")
    plt.close()
    
    # 【追加】 トライアル毎のリスク（Gap）の推移グラフ (ユーザー要望)
    plt.figure(figsize=(12, 6))
    plt.plot(df_history['trial'], df_history['gap'], marker='o', linestyle='-', color='red', label='Perception Gap (Risk)')
    plt.xlabel('Trial Number')
    plt.ylabel('Risk (Perception Gap)')
    plt.title(f'Optuna Optimization History: Risk over Trials (Scenario {scenario_name})')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    # ユーザーの指定名に合わせて optuna_worst_XXX_history_plot に保存
    plt.savefig(f"results/optuna_worst_{scenario_name}_history_plot.png")
    plt.close()
    
    print("\nBest Trial (Weather Edge Case):")
    print(f"  Max Perception Gap Score: {study.best_trial.value:.4f}")
    print(f"  Parameters: {study.best_trial.params}")
    
    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA/Mock ADAS Optuna Weather search")
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials (default: 5)")
    parser.add_argument("--sampler", type=str, default="TPE", choices=["TPE", "Random"], help="Sampler (default: TPE)")
    parser.add_argument("--scenario", type=str, default="sequence", choices=["A", "B", "C", "D", "E", "sequence"], 
                        help="Scenario skeleton: A (CPNA), B (CCRb), C (CCFtap), D (AVOID), E (RLI), sequence (all dynamically)")
    parser.add_argument("--demo", action="store_true", help="Run in mock/demo mode without CARLA server")
    args = parser.parse_args()
    
    run_optuna_search(n_trials=args.trials, sampler_name=args.sampler, scenario_name=args.scenario, demo_mode=args.demo)
