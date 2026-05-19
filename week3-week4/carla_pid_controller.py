# -*- coding: utf-8 -*-
"""
2-B. CARLAへの制御関数の組み込み
このスクリプトは、自分で調整したPIDの出力数値を、
実際のCARLAシミュレーター上のステアリング量、スロットル、ブレーキの数値に連動させるクラスです。
積分項のオーバーフロー（ワインドアップ）を防ぐアンチワインドアップ機能を実装しています。

※ 実行するにはCARLAシミュレータが起動しており、carlaモジュールがインストールされている必要があります。
"""

import math
import time

class CarlaPIDController:
    """
    CARLAの車両をPIDで制御するためのコントローラークラス。
    縦方向（速度）用と、横方向（ステアリング）用の2つのPIDを組み合わせて
    最終的な carla.VehicleControl を出力します。
    """
    def __init__(self, vehicle, dt=0.05):
        self.vehicle = vehicle
        self.dt = dt
        
        # 縦方向用PID (スロットル/ブレーキ) - 加速・減速を管理
        # 初期ゲイン: 速度追従用
        self.lon_kp = 1.0
        self.lon_ki = 0.1
        self.lon_kd = 0.05
        self.lon_integral = 0.0
        self.lon_prev_error = 0.0
        
        # 横方向用PID (ステアリング) - 左右の旋回を管理
        # 初期ゲイン: 車線維持用
        self.lat_kp = 0.8
        self.lat_ki = 0.02
        self.lat_kd = 0.1
        self.lat_integral = 0.0
        self.lat_prev_error = 0.0

    def reset(self):
        """コントローラーの状態（積分項、前回誤差）を初期化する"""
        self.lon_integral = 0.0
        self.lon_prev_error = 0.0
        self.lat_integral = 0.0
        self.lat_prev_error = 0.0

    def run_step(self, target_speed, target_steering_angle):
        """
        目標速度(m/s)と目標ステアリング角(rad)を与えて、CARLAの制御コマンドを取得する
        """
        # 現在の車両の状態を取得
        velocity = self.vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # ---------------------------------------------------------
        # 1. 縦方向（速度）のPID制御
        # ---------------------------------------------------------
        lon_error = target_speed - current_speed
        
        # 積分項の計算
        self.lon_integral += lon_error * self.dt
        
        # アンチワインドアップ (積分値が過剰に溜まるのを防ぐクランプ処理)
        # スロットル・ブレーキの指令限界(1.0 / -1.0)に対応させて積分項の影響度を抑える
        if self.lon_ki != 0.0:
            max_lon_int = 1.0 / self.lon_ki
            self.lon_integral = max(-max_lon_int, min(max_lon_int, self.lon_integral))
            
        lon_derivative = (lon_error - self.lon_prev_error) / self.dt
        
        # 出力は加速度のようなもの（プラスなら加速、マイナスなら減速）
        lon_output = (self.lon_kp * lon_error) + (self.lon_ki * self.lon_integral) + (self.lon_kd * lon_derivative)
        self.lon_prev_error = lon_error

        # ---------------------------------------------------------
        # 2. 横方向（ステアリング）のPID制御
        # ---------------------------------------------------------
        # ※ 実運用では、target_steering_angle ではなく、「目標経路と現在の進行方向との角度差や横ズレ」をエラーとして扱います。
        lat_error = target_steering_angle  # ここでは単純化のため、目標角度そのものを偏差の指標として扱う
        
        # 積分項の計算
        self.lat_integral += lat_error * self.dt
        
        # アンチワインドアップ (ステアリング限界 1.0 / -1.0 に対応して積分項をクランプ)
        if self.lat_ki != 0.0:
            max_lat_int = 1.0 / self.lat_ki
            self.lat_integral = max(-max_lat_int, min(max_lat_int, self.lat_integral))
            
        lat_derivative = (lat_error - self.lat_prev_error) / self.dt
        
        lat_output = (self.lat_kp * lat_error) + (self.lat_ki * self.lat_integral) + (self.lat_kd * lat_derivative)
        self.lat_prev_error = lat_error

        # ---------------------------------------------------------
        # 3. PID出力をCARLAの入力フォーマットに変換
        # ---------------------------------------------------------
        import carla
        control = carla.VehicleControl()
        
        # ステアリングは -1.0(左) 〜 1.0(右) の範囲に制限
        control.steer = max(-1.0, min(1.0, lat_output))
        
        # 縦方向出力がプラスならアクセル、マイナスならブレーキ
        if lon_output >= 0.0:
            control.throttle = max(0.0, min(1.0, lon_output))
            control.brake = 0.0
        else:
            control.throttle = 0.0
            # 急ブレーキによるジャークを防ぐため、ブレーキ値も0.0〜1.0にマッピング
            control.brake = max(0.0, min(1.0, abs(lon_output)))
            
        return control

# === 使用例 (擬似コード) ===
if __name__ == "__main__":
    print("CARLA Controller module ready.")
    print("CARLAシミュレータ上で実行する場合の組み込み例は以下のようになります：")
    
    """
    import carla
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    
    # 車両をスポーン
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    # コントローラーの初期化
    controller = CarlaPIDController(vehicle)
    
    try:
        while True:
            # 例: 目標速度10m/s、直進(ステアリング0)
            control = controller.run_step(target_speed=10.0, target_steering_angle=0.0)
            
            # 算出されたスロットル・ブレーキ・ステアを車両に適用
            vehicle.apply_control(control)
            
            time.sleep(0.05)
    finally:
        vehicle.destroy()
    """
