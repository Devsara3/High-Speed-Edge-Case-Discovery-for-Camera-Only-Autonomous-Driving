import optuna
import pandas as pd
import time
import os
import cv2
from carla_mock import MockCarlaEnv
from evaluator import YoloEvaluator

def run_optimization(env, evaluator, n_trials=50, sampler_name='TPE'):
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

    study = optuna.create_study(direction="minimize", sampler=sampler)

    # トライアルごとの実行時間と結果を記録するリスト
    history = []
    
    # 探索中の最も悪い（エッジケースとしてのベスト）スコアを追跡
    best_score_so_far = float('inf')
    # 参考: 最も良く認識できた画像のスコア
    worst_score_so_far = -1

    def objective(trial):
        nonlocal best_score_so_far, worst_score_so_far
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

        # 評価（検出数と信頼度の合計、および描画済み画像を取得）
        count, conf, annotated_img = evaluator.evaluate(img, return_image=True)
        
        # 目的関数スコアの計算
        # 検出数が少ないほど、エッジケース（危険な状態）とする。
        # 検出数が同じ場合は信頼度が低い方をより危険とするためのペナルティ項を追加。
        score = count + (conf / 100.0)
        
        # 画像の保存ロジック
        trial_img_path = f"results/trial_{trial.number}_{sampler_name}.jpg"
        
        # エッジケース（スコアが最小）が更新された場合
        if score < best_score_so_far:
            best_score_so_far = score
            cv2.imwrite(f"results/edge_case_worst_{sampler_name}.jpg", annotated_img)
            
        # 逆に最もよく認識できた状態（スコアが最大）を記録
        if score > worst_score_so_far:
            worst_score_so_far = score
            cv2.imwrite(f"results/edge_case_best_{sampler_name}.jpg", annotated_img)

        elapsed_time = time.time() - start_time
        
        # 履歴に記録
        history.append({
            "trial": trial.number,
            "sun_altitude_angle": sun_altitude_angle,
            "precipitation": precipitation,
            "fog_density": fog_density,
            "detected_count": count,
            "total_confidence": conf,
            "score": score,
            "elapsed_time_sec": elapsed_time
        })

        return score

    # 最適化の実行
    study.optimize(objective, n_trials=n_trials)
    
    print("\nBest Trial:")
    print(f"  Score (Minimized): {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

    return study, pd.DataFrame(history)

if __name__ == "__main__":
    # モック環境とエバリュエータの初期化
    base_image_path = "base_image.png"
    env = MockCarlaEnv(base_image_path)
    evaluator = YoloEvaluator()
    
    os.makedirs("results", exist_ok=True)

    # 1. TPE (提案手法) による探索
    study_tpe, df_tpe = run_optimization(env, evaluator, n_trials=50, sampler_name='TPE')
    df_tpe.to_csv("results/history_tpe.csv", index=False)
    
    # 2. ランダム探索 (ベースライン) との比較
    study_random, df_random = run_optimization(env, evaluator, n_trials=50, sampler_name='Random')
    df_random.to_csv("results/history_random.csv", index=False)
    
    print("\nOptimization completed. Results saved in 'results' directory.")
