import os
import glob
import pandas as pd
import numpy as np

def get_mu(obj_class):
    obj_class = str(obj_class).lower()
    if 'pedestrian' in obj_class:
        return 1.8
    elif 'car' in obj_class or 'vehicle' in obj_class:
        return 1.0
    elif 'construction' in obj_class or 'obstacle' in obj_class:
        return 1.5
    elif 'traffic_light' in obj_class or 'red' in obj_class:
        return 2.0
    return 1.0

def load_weights(weights_csv):
    """
    Load Phase 3 weights CSV.
    Expected columns: precipitation, fog, w_lidar, w_stereo, w_camera, w_ai
    """
    if not os.path.exists(weights_csv):
        print(f"[Warning] {weights_csv} not found. Using uniform weights.")
        return None
    
    df = pd.read_csv(weights_csv)
    # create a mapping dictionary based on rounded precipitation and fog (e.g., 10% bins)
    weights_map = {}
    for _, row in df.iterrows():
        # Round to nearest 10
        p_bin = int(round(row.get('precipitation', 0) / 10.0) * 10)
        f_bin = int(round(row.get('fog', 0) / 10.0) * 10)
        weights_map[(p_bin, f_bin)] = {
            'w_lidar': row.get('w_lidar', 0.25),
            'w_stereo': row.get('w_stereo', 0.25),
            'w_camera': row.get('w_camera', 0.25),
            'w_ai': row.get('w_ai', 0.25)
        }
    return weights_map

def get_weights_for_weather(weights_map, precipitation, fog):
    if weights_map is None:
        return {'w_lidar': 0.25, 'w_stereo': 0.25, 'w_camera': 0.25, 'w_ai': 0.25}
    
    p_bin = int(round(precipitation / 10.0) * 10)
    f_bin = int(round(fog / 10.0) * 10)
    
    # Try exact match, otherwise fallback to closest or uniform
    if (p_bin, f_bin) in weights_map:
        return weights_map[(p_bin, f_bin)]
    else:
        # Fallback if not found in map
        return {'w_lidar': 0.25, 'w_stereo': 0.25, 'w_camera': 0.25, 'w_ai': 0.25}

def process_trial_log(log_path, weights_map):
    """
    Process a single trial log.
    Expected columns in log:
    precipitation, fog, sun_altitude, d_lidar, d_stereo, d_camera, d_ai, v_approach, class
    """
    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        print(f"[Error] Failed to read {log_path}: {e}")
        return None

    if df.empty:
        return None

    max_risk = -float('inf')
    best_conditions = None

    for _, row in df.iterrows():
        precipitation = row.get('precipitation', 0.0)
        fog = row.get('fog', 0.0)
        sun_altitude = row.get('sun_altitude', 45.0)
        
        # Parse distances, default to large number if missing
        d_lidar = float(row.get('d_lidar', 100.0))
        d_stereo = float(row.get('d_stereo', 100.0))
        d_camera = float(row.get('d_camera', 100.0))
        d_ai = float(row.get('d_ai', 100.0))
        
        v_approach = float(row.get('v_approach', 0.0))
        obj_class = row.get('class', 'car')
        
        # Get weights for current weather
        weights = get_weights_for_weather(weights_map, precipitation, fog)
        
        # Blend distance
        d_fusion = (weights['w_lidar'] * d_lidar + 
                    weights['w_stereo'] * d_stereo + 
                    weights['w_camera'] * d_camera + 
                    weights['w_ai'] * d_ai)
        
        # Calculate Risk
        mu = get_mu(obj_class)
        K = 10.0
        epsilon = 0.01
        
        # If approach velocity is negative (moving away), risk is typically minimal or zero. 
        # Using max(0, v_approach) handles it safely.
        v_app_clamped = max(0.0, v_approach)
        
        risk_t = (K * mu * v_app_clamped) / ((d_fusion ** 2) + epsilon)
        
        if risk_t > max_risk:
            max_risk = risk_t
            best_conditions = {
                'precipitation': precipitation,
                'fog': fog,
                'sun_altitude': sun_altitude
            }
            
    if best_conditions is None:
        return None
        
    return {
        'max_risk': max_risk,
        'precipitation': best_conditions['precipitation'],
        'fog': best_conditions['fog'],
        'sun_altitude': best_conditions['sun_altitude']
    }

def main():
    print("=== Phase 4: Critical Edge Case Discovery ===")
    weights_csv = 'optimized_weather_weights.csv'
    weights_map = load_weights(weights_csv)
    
    # Read all csvs in current directory except the weights file and results file
    log_files = [f for f in glob.glob('*.csv') if f not in [weights_csv, 'phase4_results.csv']]
    
    if not log_files:
        print("[Info] No trial log CSV files found in the current directory.")
        print("Please manually place the trial CSV logs here and run again.")
        return

    results = []
    
    for log_file in log_files:
        res = process_trial_log(log_file, weights_map)
        if res is not None:
            res['trial_file'] = log_file
            results.append(res)
            
    if not results:
        print("[Info] No valid data found in logs.")
        return
        
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv('phase4_results.csv', index=False)
    print(f"[Info] Saved summary of {len(results)} trials to phase4_results.csv")
    
    # Find global maximum risk
    worst_case = results_df.loc[results_df['max_risk'].idxmax()]
    
    print("\n====================================================")
    print(" 【臨界エッジケース（Critical Edge Case）特定結果】")
    print("====================================================")
    print(f"降水量: [ {worst_case['precipitation']:.1f}% ] 以上、"
          f"霧（FOG）: [ {worst_case['fog']:.1f}% ] 以上、"
          f"日射角（SUN_ALTITUDE）: [ {worst_case['sun_altitude']-5:.1f}度〜{worst_case['sun_altitude']+5:.1f}度 ] のドメインにおいて、")
    print(f"システム全体の危険度が最大値（Risk Score: {worst_case['max_risk']:.4f}）に達した。")
    print(f"(記録元ファイル: {worst_case['trial_file']})")
    print("====================================================\n")

if __name__ == '__main__':
    main()
