# -*- coding: utf-8 -*-
"""
CARLA PID Control Comparison Runner
このスクリプトは、同一のシミュレーション環境において、縦・横方向のPIDパラメータ設定値の組み合わせによる
挙動への影響を個別に分離して検証・比較するためのオートメーションスクリプトです。

以下の3パターンを自動で連続実行し、それぞれの結果（動画および解析写真）を個別に保存します。
1. パターンA（両方最適）: 縦方向（加減速）も横方向（ステアリング）も正しくチューニングされた状態
2. パターンB（縦だけ不良）: 横方向は正常ですが、縦方向（アクセル）が弱すぎて目標速度に届かない状態
3. パターンC（横だけ不良）: 縦方向は正常ですが、横方向（ハンドル）が弱すぎてカーブを曲がれず道路を外れる状態
"""

import subprocess
import sys
import os
import time

def run_pattern(pattern_name, lon_gains, lat_gains, target_speed=10.0):
    kp_lon, ki_lon, kd_lon = lon_gains
    kp_lat, ki_lat, kd_lat = lat_gains
    
    print("====================================================")
    print(f" 実行中: {pattern_name}")
    print(f" 目標速度: {target_speed} m/s (36 km/h)")
    print(f" 縦方向PID (速度用)     : Kp={kp_lon}, Ki={ki_lon}, Kd={kd_lon}")
    print(f" 横方向PID (操舵・CTE用): Kp={kp_lat}, Ki={ki_lat}, Kd={kd_lat}")
    print("====================================================")
    
    cmd = [
        sys.executable,
        "carla_integration_demo.py",
        "--target-speed", str(target_speed),
        "--kp-lon", str(kp_lon),
        "--ki-lon", str(ki_lon),
        "--kd-lon", str(kd_lon),
        "--kp-lat", str(kp_lat),
        "--ki-lat", str(ki_lat),
        "--kd-lat", str(kd_lat)
    ]
    
    try:
        process = subprocess.Popen(cmd, cwd=os.path.dirname(__file__))
        process.wait()
        if process.returncode == 0:
            print(f"\n[成功] {pattern_name} のシミュレーションが完了しました。\n")
        else:
            print(f"\n[エラー] {pattern_name} が終了コード {process.returncode} で異常終了しました。\n")
    except Exception as e:
        print(f"\n[例外] {pattern_name} の実行中にエラーが発生しました: {e}\n")

def get_suffix(lon, lat):
    return f"lon_{lon[0]}_{lon[1]}_{lon[2]}_lat_{lat[0]}_{lat[1]}_{lat[2]}"

def main():
    print("====================================================")
    print(" CARLA PID 縦・横 影響分離比較オートメーションランナー")
    print("====================================================")
    print("このランナーは、縦方向（加減速）と横方向（ハンドル）のパラメータが")
    print("それぞれ「うまくいく場合」と「うまくいかない場合」の影響を個別に比較するため、")
    print("以下の3つのパターンを同じ条件下で連続して実行します。\n")
    
    # ---------------------------------------------------------
    # パターン設定
    # ---------------------------------------------------------
    # パターンA: 【両方最適】 (Ego車はスムーズに加速し、車線の中央をトレースし、障害物手前で綺麗に停止)
    patternA_name = "パターンA (縦: 最適 / 横: 最適)"
    patternA_lon = (1.2, 0.1, 0.05)
    patternA_lat = (0.8, 0.02, 0.1)
    
    # パターンB: 【縦だけ不良・横は最適】 (ハンドル操作は正確だが、アクセルが弱すぎて目標速度にまったく届かない)
    patternB_name = "パターンB (縦: 不良 / 横: 最適)"
    patternB_lon = (0.1, 0.0, 0.0)  # Kpが小さすぎて加速が非常に遅い
    patternB_lat = (0.8, 0.02, 0.1) # 横は最適のまま
    
    # パターンC: 【縦は最適・横だけ不良】 (加速はスムーズだが、ハンドルが弱すぎてカーブを曲がりきれずにコースアウト)
    patternC_name = "パターンC (縦: 最適 / 横: 不良)"
    patternC_lon = (1.2, 0.1, 0.05) # 縦は最適のまま
    patternC_lat = (0.1, 0.0, 0.0)  # Kpが小さすぎてハンドルがほとんど切れない
    
    # ---------------------------------------------------------
    # 実行フェーズ
    # ---------------------------------------------------------
    # パターンAを実行
    run_pattern(patternA_name, patternA_lon, patternA_lat)
    print("アクターのクリーンアップとCARLAの初期化を待機中 (3秒)...")
    time.sleep(3.0)
    
    # パターンBを実行
    run_pattern(patternB_name, patternB_lon, patternB_lat)
    print("アクターのクリーンアップとCARLAの初期化を待機中 (3秒)...")
    time.sleep(3.0)
    
    # パターンCを実行
    run_pattern(patternC_name, patternC_lon, patternC_lat)
    
    # ---------------------------------------------------------
    # 結果サマリー表示
    # ---------------------------------------------------------
    suffixA = get_suffix(patternA_lon, patternA_lat)
    suffixB = get_suffix(patternB_lon, patternB_lat)
    suffixC = get_suffix(patternC_lon, patternC_lat)
    
    print("====================================================")
    print(" 比較シミュレーション完了！結果ファイル一覧")
    print("====================================================")
    print("生成された以下の写真（プロット）と動画を並べて開き、制御への影響を比較してください。\n")
    
    print(f"🎬 【{patternA_name} の出力（両方成功モデル）】")
    print(f"  - 動画: carla_run_PID_{suffixA}_recording.avi")
    print(f"  - グラフ写真 (PID挙動): carla_pid_tuning_{suffixA}_results.png")
    print(f"  - グラフ写真 (全センサー): carla_sensor_analysis_{suffixA}.png\n")
    
    print(f"🎬 【{patternB_name} の出力（縦方向のみ不良・低加速モデル）】")
    print(f"  - 動画: carla_run_PID_{suffixB}_recording.avi")
    print(f"  - グラフ写真 (PID挙動): carla_pid_tuning_{suffixB}_results.png")
    print(f"  - グラフ写真 (全センサー): carla_sensor_analysis_{suffixB}.png\n")
    
    print(f"🎬 【{patternC_name} の出力（横方向のみ不良・コースアウトモデル）】")
    print(f"  - 動画: carla_run_PID_{suffixC}_recording.avi")
    print(f"  - グラフ写真 (PID挙動): carla_pid_tuning_{suffixC}_results.png")
    print(f"  - グラフ写真 (全センサー): carla_sensor_analysis_{suffixC}.png")
    print("====================================================")

if __name__ == '__main__':
    main()
