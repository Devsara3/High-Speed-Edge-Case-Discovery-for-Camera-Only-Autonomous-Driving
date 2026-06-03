import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_weights_for_weather(df_group):
    """
    特定の天候条件下におけるセンサーの最適重みを計算する関数
    """
    # 1. 計算に使用する物理量（距離データ）をNumPy配列として抽出
    d_gt = df_group['dist_gt'].to_numpy()
    d_lidar = df_group['dist_lidar'].to_numpy()
    d_stereo = df_group['dist_stereo'].to_numpy()
    d_camera = df_group['dist_camera'].to_numpy()
    d_ai = df_group['dist_ai'].to_numpy()
    
    # 2. 目的関数の定義 (真値とフュージョン距離の二乗誤差の合計)
    def objective_function(weights):
        w_lidar, w_stereo, w_camera, w_ai = weights
        
        # 加重線形結合（Late Fusion）によるフュージョン距離の計算
        d_fusion = (w_lidar * d_lidar + 
                    w_stereo * d_stereo + 
                    w_camera * d_camera + 
                    w_ai * d_ai)
        
        # 二乗誤差（残差平方和）
        loss = np.sum((d_gt - d_fusion) ** 2)
        return loss

    # 3. 制約条件と境界値の定義
    # 各重みの範囲を 0.0 〜 1.0 に制限 (Bounds)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    
    # 重みの合計を 1.0 にする制約 (Constraint: sum(w) - 1.0 = 0)
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}
    
    # 4. 初期値の設定 (均等配分: 各25%)
    initial_weights = [0.25, 0.25, 0.25, 0.25]
    
    # 5. SLSQP（逐次二次計画法）による最適化の実行
    result = minimize(
        objective_function, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    if result.success:
        return result.x  # [w_lidar, w_stereo, w_camera, w_ai]
    else:
        return initial_weights  # 最適化失敗時は初期値をフォールバック

def main():
    csv_path = 'experiment_log.csv'
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} が見つかりません。データ収集を先に完了させてください。")
        return

    # データの読み込み
    print(f"[INFO] ログファイル {csv_path} を読み込み中...")
    df = pd.read_csv(csv_path)
    
    # 不正な値（センサーロスト等のinfやNaN）を持つ行を事前に排除
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['dist_gt', 'dist_lidar', 'dist_stereo', 'dist_camera', 'dist_ai'])

    # 天候条件（雨・霧）を5%刻みや10%刻みなどの「区切り（ビン）」に丸めてグループ化
    # ※丸めないと完全一致するログが少なすぎて最適化が安定しないため
    df['precip_grid'] = np.round(df['precipitation'] / 10.0) * 10.0
    df['fog_grid'] = np.round(df['fog'] / 10.0) * 10.0
    
    unique_weathers = df.groupby(['precip_grid', 'fog_grid'])
    
    optimization_results = []
    
    print("--- 天候別のセンサー重み最適化を開始 ---")
    for (precip, fog), group in unique_weathers:
        # サンプル数が少なすぎる天候は信頼性が低いためスキップ（例: 20コマ未満）
        if len(group) < 20:
            continue
            
        # 最適化関数の呼び出し
        w_lidar, w_stereo, w_camera, w_ai = optimize_weights_for_weather(group)
        
        optimization_results.append({
            'precipitation': precip,
            'fog': fog,
            'w_lidar': w_lidar,
            'w_stereo': w_stereo,
            'w_camera': w_camera,
            'w_ai': w_ai,
            'sample_count': len(group)
        })
        print(f"天候 -> 雨:{precip:5.1f}%, 霧:{fog:5.1f}% | 最適重み -> LiDAR:{w_lidar:.2f}, Stereo:{w_stereo:.2f}, Camera:{w_camera:.2f}, AI:{w_ai:.2f} (サンプル数: {len(group)})")

    # 結果をデータフレームにして保存
    result_df = pd.DataFrame(optimization_results)
    output_path = 'optimized_weather_weights.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] すべての天候パターンに対する最適化重みマップを {output_path} にエクスポートしました！")

if __name__ == '__main__':
    main()
