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
        
        try:
            # 天候適用
            experiment.set_weather_params(sun_altitude_angle, precipitation, fog_density)
            
            # シナリオの実行
            if scenario_name == 'sequence':
                experiment.run_sequence()
            else:
                experiment.run_experiment(scenario_name)
                
            # 最大認識ギャップを目的関数値として取得
            max_gap = experiment.get_max_gap()
            
            # 最悪結果の更新と可視化グラフ保存
            if max_gap > worst_gap:
                worst_gap = max_gap
                experiment.visualize_and_save(f"results/optuna_worst_{scenario_name}.png")
                print(f"\n--> [NEW WORST WEATHER DISCOVERD] Gap: {max_gap:.4f}")
                print(f"    Params: Sun Alt={sun_altitude_angle:.2f}, Rain={precipitation:.2f}, Fog={fog_density:.2f}\n")
                
        finally:
            experiment.shutdown()
            
        elapsed_time = time.time() - start_time
        
        history.append({
            "trial": trial.number,
            "sun_altitude_angle": sun_altitude_angle,
            "precipitation": precipitation,
            "fog_density": fog_density,
            "gap": max_gap,
            "elapsed_time_sec": elapsed_time
        })
        
        return max_gap

    study.optimize(objective, n_trials=n_trials)
    
    # 歴史データをCSVに記録
    df_history = pd.DataFrame(history)
    df_history.to_csv(f"results/optuna_history_{scenario_name}_{sampler_name}.csv", index=False)
    
    print("\nBest Trial (Weather Edge Case):")
    print(f"  Max Perception Gap Score: {study.best_trial.value:.4f}")
    print(f"  Parameters: {study.best_trial.params}")
    
    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARLA/Mock ADAS Optuna Weather search")
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials (default: 5)")
    parser.add_argument("--sampler", type=str, default="TPE", choices=["TPE", "Random"], help="Sampler (default: TPE)")
    parser.add_argument("--scenario", type=str, default="sequence", choices=["A", "B", "C", "D", "sequence"], 
                        help="Scenario skeleton: A (CPNA), B (CCRb), C (CCFtap), D (AVOID), sequence (all dynamically)")
    parser.add_argument("--demo", action="store_true", help="Run in mock/demo mode without CARLA server")
    args = parser.parse_args()
    
    run_optuna_search(n_trials=args.trials, sampler_name=args.sampler, scenario_name=args.scenario, demo_mode=args.demo)
