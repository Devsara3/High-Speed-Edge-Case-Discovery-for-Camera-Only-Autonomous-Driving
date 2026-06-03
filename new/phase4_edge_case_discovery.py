import os
import glob
import numpy as np
import pandas as pd

def calculate_transparent_risk(df_merged, K=10.0, epsilon=0.01):
    """
    透明性を完全確保し、全ての計算中間パラメータを明記してリスクスコア R(t) を算出する
    """
    # 1. 認識ラベル（scenario_type）に応じたハザード属性係数 μ のマッピング
    mu_map = {'A': 1.8, 'B': 1.0, 'C': 1.0, 'D': 1.5, 'E': 2.0}
    df_merged['mu_factor'] = df_merged['scenario_type'].map(mu_map).fillna(1.0)
    
    # 2. 統合距離 d_fusion(t) の算出プロセスの可視化
    df_merged['d_fusion'] = (
        df_merged['w_lidar'] * df_merged['dist_lidar'] +
        df_merged['w_stereo'] * df_merged['dist_stereo'] +
        df_merged['w_camera'] * df_merged['dist_camera'] +
        df_merged['w_ai'] * df_merged['dist_ai'] +
        df_merged['w_hsi'] * df_merged['dist_hsi']
    )
    
    # 3. 遠ざかるハザードへの最低リスク保証 (0.1)
    df_merged['v_app_clamped'] = np.where(df_merged['v_approach'] > 0, df_merged['v_approach'], 0.1)
    
    # 4. リスクスコア R(t) の数理計算
    # 式の透明性を確認できるよう、分子 (numerator) と分母 (denominator) もカラムに分離して保存
    df_merged['risk_numerator'] = K * df_merged['mu_factor'] * df_merged['v_app_clamped']
    df_merged['risk_denominator'] = (df_merged['d_fusion'] ** 2) + epsilon
    
    df_merged['risk_score'] = df_merged['risk_numerator'] / df_merged['risk_denominator']
    
    return df_merged

def generate_mock_data():
    """
    Verification Plan に基づくモックデータの自動生成
    """
    os.makedirs('logs', exist_ok=True)
    
    # 1. Phase 3 のダミー重みデータ作成
    weights_data = []
    for p in range(0, 101, 10):
        for f in range(0, 101, 10):
                w_ai = (p + f) / 200.0 if (p + f) > 0 else 0.2
                w_hsi = 0.1
                if -30 <= s <= 30: # 日差しが強いときはLiDAR等への依存を増やすモック
                    w_ai = min(w_ai + 0.1, 0.5)
                    w_hsi = min(w_hsi + 0.1, 0.5)
                remaining = 1.0 - w_ai - w_hsi
                weights_data.append({
                    'precipitation': float(p), 'fog': float(f), 'sun_altitude': float(s),
                    'w_lidar': remaining * 0.5, 'w_stereo': remaining * 0.3,
                    'w_camera': remaining * 0.2, 'w_ai': w_ai, 'w_hsi': w_hsi
                })
    pd.DataFrame(weights_data).to_csv('optimized_weather_weights.csv', index=False)
    
    # 2. Phase 2 のダミー走行ログ（50回分）の作成
    scenarios = ['A', 'B', 'C', 'D', 'E']
    for i in range(50):
        p = np.random.choice(range(0, 101, 10))
        f = np.random.choice(range(0, 101, 10))
        sun = np.random.uniform(-90, 90)
        scenario = np.random.choice(scenarios)
        
        ticks = np.arange(1, 101)
        # 半分のデータは接近、半分のデータは遠ざかる（安全弁のテスト用）
        if i % 2 == 0:
            dist_gt = np.linspace(40, 0.5, 100)
            v_approach = np.random.uniform(5, 15, 100)
        else:
            dist_gt = np.linspace(5.0, 30.0, 100)
            v_approach = np.linspace(-2.0, -10.0, 100)
            
        noise_factor = (p + f) / 50.0
        df_trial = pd.DataFrame({
            'tick': ticks, 'precipitation': p, 'fog': f, 'sun_altitude': sun,
            'scenario_type': scenario, 'v_approach': v_approach, 'dist_gt': dist_gt,
            'dist_lidar': dist_gt + np.random.normal(0, 0.1 * noise_factor, 100),
            'dist_stereo': dist_gt + np.random.normal(0, 0.5 * noise_factor, 100),
            'dist_camera': dist_gt + np.random.normal(0, 1.2 * noise_factor, 100),
            'dist_ai': dist_gt + np.random.normal(0, 0.2 * noise_factor, 100),
            'dist_hsi': dist_gt + np.random.normal(0, 0.15 * noise_factor, 100)
        })
        if p >= 80 and f >= 80 and -10 <= sun <= 10:
            df_trial['v_approach'] *= 2.5
            df_trial['dist_stereo'] += 15.0 
            
        df_trial.to_csv(f'logs/trial_{i:03d}.csv', index=False)
    print("[SUCCESS] Verification用のモックデータ（重みCSV 1本 & 走行ログ 50本）を生成しました。")

def main():
    log_files = glob.glob('logs/trial_*.csv')
    if os.path.exists('results/experiment_log.csv'):
        log_files.append('results/experiment_log.csv')
        
    if not os.path.exists('optimized_weather_weights.csv') or len(log_files) == 0:
        generate_mock_data()
        log_files = glob.glob('logs/trial_*.csv')
        
    df_weights = pd.read_csv('optimized_weather_weights.csv')
    
    all_trials_summary = []
    all_timesteps_list = []
    
    print(f"[INFO] {len(log_files)} 本のログから数理パラメータを全出力してマージ中...")
    for file in log_files:
        df_log = pd.read_csv(file)
        
        df_log['precip_grid'] = np.round(df_log['precipitation'] / 10.0) * 10.0
        df_log['fog_grid'] = np.round(df_log['fog'] / 10.0) * 10.0
        df_log['sun_grid'] = np.round(df_log['sun_altitude'] / 15.0) * 15.0
        
        df_merged = pd.merge(
            df_log, df_weights, 
            left_on=['precip_grid', 'fog_grid', 'sun_grid'], 
            right_on=['precipitation', 'fog', 'sun_altitude'], 
            suffixes=('', '_w')
        )
        
        if df_merged.empty:
            continue
            
        df_transparent = calculate_transparent_risk(df_merged)
        all_timesteps_list.append(df_transparent)
        
        max_idx = df_transparent['risk_score'].idxmax()
        worst_row = df_transparent.loc[max_idx]
        
        all_trials_summary.append({
            'scenario_type': worst_row['scenario_type'],
            'precipitation': worst_row['precipitation'],
            'fog': worst_row['fog'],
            'sun_altitude': worst_row['sun_altitude'],
            'max_risk_score': worst_row['risk_score']
        })
        
    if not all_timesteps_list:
        return
        
    df_final_archive = pd.concat(all_timesteps_list)
    df_summary = pd.DataFrame(all_trials_summary)
    
    output_columns = [
        'tick', 'scenario_type', 'precipitation', 'fog', 'sun_altitude',
        'dist_gt', 'dist_lidar', 'dist_stereo', 'dist_camera', 'dist_ai', 'dist_hsi',
        'w_lidar', 'w_stereo', 'w_camera', 'w_ai', 'w_hsi',
        'd_fusion', 'v_approach', 'v_app_clamped',
        'mu_factor', 'risk_numerator', 'risk_denominator', 'risk_score'
    ]
    existing_cols = [col for col in output_columns if col in df_final_archive.columns]
    
    # plot_results.py が読み込むファイル名に上書き
    output_csv_path = 'fused_risk_timeseries.csv'
    df_final_archive[existing_cols].to_csv(output_csv_path, index=False)
    print(f"\n[SUCCESS] リスク計算の透明性を検証するための全ログを {output_csv_path} に保存しました！")
    
    print("\n--- 監査用データサンプル（v_approachがマイナス時の挙動） ---")
    neg_v_sample = df_final_archive[df_final_archive['v_approach'] < 0]
    if not neg_v_sample.empty:
        print(neg_v_sample[['tick', 'dist_gt', 'v_approach', 'v_app_clamped', 'risk_score']].head(5))
    else:
        print("（※マイナスのv_approachデータはありませんでした）")
    
    print("\n======================================================================================")
    print("                 [PHASE 4] CRITICAL EDGE CASE REPORT (PER SCENARIO)                   ")
    print("======================================================================================")
    
    scenarios_found = sorted(df_summary['scenario_type'].unique())
    
    for scen in scenarios_found:
        df_scen = df_summary[df_summary['scenario_type'] == scen]
        worst_case = df_scen.sort_values(by='max_risk_score', ascending=False).iloc[0]
        sun_min = np.floor(worst_case['sun_altitude'] / 5.0) * 5.0
        sun_max = sun_min + 5.0
        
        print(f"【シナリオ {scen} の最悪エッジケース】")
        print(f"  - 降水量: [ {worst_case['precipitation']:.1f}% ] 以上、霧（FOG）: [ {worst_case['fog']:.1f}% ] 以上")
        print(f"  - 日射角: [ {sun_min:.1f}度 〜 {sun_max:.1f}度 ]")
        print(f"  - Max Risk Score: {worst_case['max_risk_score']:.2f}")
        print("-" * 86)
    print()

if __name__ == '__main__':
    main()
