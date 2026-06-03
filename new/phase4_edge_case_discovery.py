import os
import glob
import numpy as np
import pandas as pd

def calculate_dynamic_risk(df_merged, K=10.0, epsilon=0.01):
    """
    数理モデルに基づき、各タイムステップのリスクスコア R(t) を算出する
    """
    # 物体クラス（scenario_type）に応じたハザード属性係数 μ のマッピング
    mu_map = {'A': 1.8, 'B': 1.0, 'C': 1.0, 'D': 1.5, 'E': 2.0}
    df_merged['mu'] = df_merged['scenario_type'].map(mu_map).fillna(1.0)
    
    # 統合距離 d_fusion(t) の算出
    df_merged['d_fusion'] = (
        df_merged['w_lidar'] * df_merged['dist_lidar'] +
        df_merged['w_stereo'] * df_merged['dist_stereo'] +
        df_merged['w_camera'] * df_merged['dist_camera'] +
        df_merged['w_ai'] * df_merged['dist_ai']
    )
    
    # リスクスコア R(t) の数理計算
    # ※接近速度がマイナス（離脱）の場合にリスクがマイナスにならないよう、0以下をカット（np.maximumを追加）
    v_app_clamped = np.maximum(0, df_merged['v_approach'])
    df_merged['risk_score'] = (K * df_merged['mu'] * v_app_clamped) / (df_merged['d_fusion']**2 + epsilon)
    return df_merged

def generate_mock_data():
    """
    Verification Plan に基づくモックデータの自動生成
    """
    os.makedirs('new/logs', exist_ok=True)
    
    # 1. Phase 3 のダミー重みデータ作成
    weights_data = []
    # ※merge時の欠落を防ぐため、10%刻み(step=10)に変更しています
    for p in range(0, 101, 10):
        for f in range(0, 101, 10):
            # 霧や雨が強いほど AI (w_ai) の重みが上がるダミー特性
            w_ai = (p + f) / 200.0 if (p + f) > 0 else 0.25
            remaining = 1.0 - w_ai
            weights_data.append({
                'precipitation': float(p), 'fog': float(f),
                'w_lidar': remaining * 0.5, 'w_stereo': remaining * 0.3,
                'w_camera': remaining * 0.2, 'w_ai': w_ai
            })
    pd.DataFrame(weights_data).to_csv('new/optimized_weather_weights.csv', index=False)
    
    # 2. Phase 2 のダミー走行ログ（50回分）の作成
    scenarios = ['A', 'B', 'C', 'D', 'E']
    for i in range(50):
        p = np.random.choice(range(0, 101, 10))
        f = np.random.choice(range(0, 101, 10))
        sun = np.random.uniform(-90, 90)
        scenario = np.random.choice(scenarios)
        
        # 1トライアルあたり100コマの時系列データ
        ticks = np.arange(1, 101)
        dist_gt = np.linspace(40, 0.5, 100) # 近づいてくるハザード
        v_approach = np.random.uniform(5, 15, 100)
        
        # 悪天候ほどセンサーが狂うノイズを付加
        noise_factor = (p + f) / 50.0
        df_trial = pd.DataFrame({
            'tick': ticks, 'precipitation': p, 'fog': f, 'sun_altitude': sun,
            'scenario_type': scenario, 'v_approach': v_approach, 'dist_gt': dist_gt,
            'dist_lidar': dist_gt + np.random.normal(0, 0.1 * noise_factor, 100),
            'dist_stereo': dist_gt + np.random.normal(0, 0.5 * noise_factor, 100),
            'dist_camera': dist_gt + np.random.normal(0, 1.2 * noise_factor, 100),
            'dist_ai': dist_gt + np.random.normal(0, 0.2 * noise_factor, 100)
        })
        # 特定の天候（例: 降水80, 霧80, 西日付近）で大クラッシュ（高リスク）が起きる仕込み
        if p >= 80 and f >= 80 and -10 <= sun <= 10:
            df_trial['v_approach'] *= 2.5
            df_trial['dist_stereo'] += 15.0 # ステレオが激しく狂う
            
        df_trial.to_csv(f'new/logs/trial_{i:03d}.csv', index=False)
    print("[SUCCESS] Verification用のモックデータ（重みCSV 1本 & 走行ログ 50本）を生成しました。")

def main():
    # ログがない場合はモックを生成
    if not os.path.exists('new/optimized_weather_weights.csv') or not glob.glob('new/logs/trial_*.csv'):
        generate_mock_data()
        
    # データの読み込み
    df_weights = pd.read_csv('new/optimized_weather_weights.csv')
    log_files = glob.glob('new/logs/trial_*.csv')
    
    all_trials_summary = []
    all_timesteps_list = [] # 視覚化コードに引き渡す用
    
    print(f"[INFO] {len(log_files)} 本の走行ログをスキャン中...")
    for file in log_files:
        df_log = pd.read_csv(file)
        
        # 重みテーブルと天候（10%刻みの丸め）でマージ
        df_log['precip_grid'] = np.round(df_log['precipitation'] / 10.0) * 10.0
        df_log['fog_grid'] = np.round(df_log['fog'] / 10.0) * 10.0
        
        df_merged = pd.merge(
            df_log, df_weights, 
            left_on=['precip_grid', 'fog_grid'], 
            right_on=['precipitation', 'fog'], 
            suffixes=('', '_w')
        )
        
        if df_merged.empty:
            continue
            
        # リスク計算
        df_calculated = calculate_dynamic_risk(df_merged)
        all_timesteps_list.append(df_calculated)
        
        # トライアルごとの Max R(t) を算出
        max_idx = df_calculated['risk_score'].idxmax()
        worst_row = df_calculated.loc[max_idx]
        
        all_trials_summary.append({
            'scenario_type': worst_row['scenario_type'],
            'precipitation': worst_row['precipitation'],
            'fog': worst_row['fog'],
            'sun_altitude': worst_row['sun_altitude'],
            'max_risk_score': worst_row['risk_score']
        })
        
    df_summary = pd.DataFrame(all_trials_summary)
    
    # すべてのタイムステップ統合データをプロット用に保存
    pd.concat(all_timesteps_list).to_csv('new/fused_risk_timeseries.csv', index=False)
    
    # 臨界エッジケースの特定（最大リスクスコアを記録した条件）
    worst_case = df_summary.sort_values(by='max_risk_score', ascending=False).iloc[0]
    
    # 日射角のドメインを範囲（±5度）としてマージ
    sun_min = np.floor(worst_case['sun_altitude'] / 5.0) * 5.0
    sun_max = sun_min + 5.0
    
    print("\n======================================================================================")
    print("                      📊 PHASE 4: CRITICAL EDGE CASE REPORT                           ")
    print("======================================================================================")
    print(f"降水量: [ {worst_case['precipitation']:.1f}% ] 以上、霧（FOG）: [ {worst_case['fog']:.1f}% ] 以上、")
    print(f"日射角（SUN_ALTITUDE）: [ {sun_min:.1f}度 〜 {sun_max:.1f}度 ] のドメインにおいて、")
    print(f"システム全体の危険度が最大値（Max Risk Score: {worst_case['max_risk_score']:.2f}）に達した。")
    print(f"※対象となった最悪のハザードシナリオ型: シナリオ {worst_case['scenario_type']}")
    print("======================================================================================\n")

if __name__ == '__main__':
    main()
