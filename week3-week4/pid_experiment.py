# -*- coding: utf-8 -*-
"""
Track 3: 自動運転の制御（2次元PIDコントローラーと自転車モデルのシミュレーション）
ロビン先生・チーム報告用のPIDチューニング＆可視化スクリプト。

このスクリプトは、2自由度の制御（縦方向：速度、横方向：ステアリング）を独立して適用し、
「キネマティック・自転車モデル（Kinematic Bicycle Model）」に基づいた2次元経路追従をシミュレーションします。
チューニングパラメータが異なる3つのケース（適切、過敏、鈍感）を同時に走行させ、
軌跡、偏差（CTE）、快適性（加加速度/Jerk）などのデータをグラフ（PNG）およびアニメーション動画（GIF）として保存します。
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. PIDコントローラークラスの定義 ---
class PIDController:
    """
    比例(P)、積分(I)、微分(D)の3つの項を計算し、制御入力にフィードバックする汎用PIDコントローラー。
    ワインドアップ（積分項の過剰累積）を防ぐためのアンチワインドアップ（積分クランプ）機能を搭載。
    """
    def __init__(self, kp, ki, kd, dt, output_limits=(None, None), anti_windup=True):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.min_out, self.max_out = output_limits
        self.anti_windup = anti_windup
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False
        
    def step(self, error):
        # P項 (比例)
        p_output = self.kp * error
        
        # I項 (積分) - アンチワインドアップの処理を適用
        self.integral += error * self.dt
        if self.anti_windup:
            # 積分項単体での限界を制限（出力限界から積分項が飽和するのを防ぐ）
            if self.ki != 0.0 and self.min_out is not None and self.max_out is not None:
                max_int = self.max_out / self.ki
                min_int = self.min_out / self.ki
                self.integral = max(min_int, min(max_int, self.integral))
        i_output = self.ki * self.integral
        
        # D項 (微分) - 初期ステップの急激な変化（スパイク）を防ぐガード
        if not self.initialized:
            self.prev_error = error
            self.initialized = True
        
        d_output = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        # PIDの合計出力
        output = p_output + i_output + d_output
        
        # 出力値の制限 (スロットル/ブレーキやステアリングの物理限界)
        if self.min_out is not None:
            output = max(self.min_out, output)
        if self.max_out is not None:
            output = min(self.max_out, output)
            
        return output

    def reset(self):
        """シミュレーション再開用に内部状態を初期化"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

# --- 2. 2次元キネマティック自転車モデルの定義 ---
class KinematicBicycleVehicle:
    """
    キネマティック・自転車モデル（Kinematic Bicycle Model）を用いた車両物理の簡略化シミュレーション。
    状態量: x, y (位置), yaw (向き/ラジアン), v (速度 m/s)
    入力値: accel (加速度 m/s^2), steer (ステアリング角 ラジアン)
    """
    def __init__(self, x=0.0, y=1.0, yaw=0.0, v=0.0, wheelbase=2.5, dt=0.05):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = wheelbase  # ホイールベース（前輪と後輪の距離、Tesla Model 3等は約2.8mだがここでは2.5mとする）
        self.dt = dt
        
        # 評価用の走行履歴記録
        self.history = {
            't': [0.0],
            'x': [self.x],
            'y': [self.y],
            'yaw': [self.yaw],
            'v': [self.v],
            'accel': [0.0],
            'steer': [0.0],
            'cte': [0.0],
            'heading_error': [0.0],
            'jerk': [0.0]
        }
        self.prev_accel = 0.0

    def update(self, accel_cmd, steer_cmd):
        """
        車両の運動方程式に基づいて状態を更新する。
        - 縦方向: 速度 v を加速度 accel_cmd で更新
        - 横方向: ステアリング角 steer_cmd をもとに旋回半径を求め、x, y, yaw を更新
        """
        # 物理限界によるクランプ
        # 加速度：ブレーキは強く (-8.0 m/s^2), アクセルは緩やかに (4.0 m/s^2)
        accel = np.clip(accel_cmd, -8.0, 4.0)
        # ステアリング角：最大45度 (pi/4 ラジアン)
        steer = np.clip(steer_cmd, -math.pi/4, math.pi/4)
        
        # 乗り心地の指標：加加速度 (Jerk = d_accel/dt)。急激な操作があると大きくなる
        jerk = (accel - self.prev_accel) / self.dt
        self.prev_accel = accel
        
        # 自転車モデルの幾何学的運動方程式
        # x_dot = v * cos(yaw)
        # y_dot = v * sin(yaw)
        # yaw_dot = v / L * tan(steer)
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        self.yaw += (self.v / self.L) * math.tan(steer) * self.dt
        self.v += accel * self.dt
        
        # 後退は行わない（前進専用の想定）
        if self.v < 0.0:
            self.v = 0.0
            
        # 角度を [-pi, pi] の範囲に正規化
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        
        return accel, steer, jerk

    def record(self, t, accel, steer, jerk, cte, heading_error):
        """現在の状態を履歴に追加"""
        self.history['t'].append(t)
        self.history['x'].append(self.x)
        self.history['y'].append(self.y)
        self.history['yaw'].append(self.yaw)
        self.history['v'].append(self.v)
        self.history['accel'].append(accel)
        self.history['steer'].append(steer)
        self.history['jerk'].append(jerk)
        self.history['cte'].append(cte)
        self.history['heading_error'].append(heading_error)

# --- 3. 軌道エラー（CTE / 角度ズレ）の計算関数 ---
def calculate_path_errors(x, y, yaw, path_x, path_y, path_yaw):
    """
    車両の現在位置 (x, y) から目標経路上の最も近いポイントを探索し、
    クロス・トラック・エラー (CTE) と方位誤差 (heading_error) を計算する。
    """
    # 全ウェイポイントに対するユークリッド距離を計算
    dx = path_x - x
    dy = path_y - y
    distances = np.sqrt(dx**2 + dy**2)
    closest_idx = np.argmin(distances)
    
    # 最も近いウェイポイントの座標と方向
    ref_x = path_x[closest_idx]
    ref_y = path_y[closest_idx]
    ref_yaw = path_yaw[closest_idx]
    
    # クロス・トラック・エラー (CTE) の算出: 
    # 車両位置と目標点のベクトルを、目標点の法線ベクトル [-sin(ref_yaw), cos(ref_yaw)] に投影
    # 左にズレている場合は正値、右にズレている場合は負値となる
    diff_x = x - ref_x
    diff_y = y - ref_y
    cte = -diff_x * math.sin(ref_yaw) + diff_y * math.cos(ref_yaw)
    
    # 方位誤差 (Heading Error)
    heading_error = ref_yaw - yaw
    # 角度を [-pi, pi] に正規化
    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
    
    return cte, heading_error, closest_idx

# --- 4. シミュレーション実行用関数 ---
def run_2d_simulation(path_x, path_y, path_yaw, target_speed, lon_gains, lat_gains, duration=18.0, dt=0.05):
    """
    指定されたPIDゲイン（縦・横）を用いて車両シミュレーションを実行する。
    """
    # 車両の初期化（少し経路からズレた位置からスタート）
    vehicle = KinematicBicycleVehicle(x=0.0, y=1.0, yaw=0.0, v=0.0, dt=dt)
    
    # 縦方向制御（速度）用PID
    kp_lon, ki_lon, kd_lon = lon_gains
    lon_pid = PIDController(kp_lon, ki_lon, kd_lon, dt, output_limits=(-8.0, 4.0))
    
    # 横方向制御（ステアリング）用PID
    # 目標ステアはラジアン表記 (-pi/4 〜 pi/4)
    kp_lat, ki_lat, kd_lat = lat_gains
    lat_pid = PIDController(kp_lat, ki_lat, kd_lat, dt, output_limits=(-math.pi/4, math.pi/4))
    
    steps = int(duration / dt)
    
    for i in range(steps):
        t = (i + 1) * dt
        
        # エラーの計算
        cte, heading_error, idx = calculate_path_errors(vehicle.x, vehicle.y, vehicle.yaw, path_x, path_y, path_yaw)
        
        # 経路の終端付近に達したら終了
        if idx >= len(path_x) - 5:
            break
            
        # 縦方向 (Longitudinal) : 速度エラーをもとにアクセル/ブレーキ量を決定
        speed_error = target_speed - vehicle.v
        accel_cmd = lon_pid.step(speed_error)
        
        # 横方向 (Lateral) : 経路エラーをもとにステアリング操舵角を決定
        # CTEが正値（左にずれている）の時、車両は右に曲がりたい（ステアリング角は負値）のでエラー入力を -cte にする
        # 方位誤差（heading_error）も補正として加える（D項と同様の効果を持ち、応答をより滑らかにする）
        # ※ ここでは純粋な「CTEに対するPID」と「方向の偏差」を直感的に組み合わせるために、
        #    steer_error = -cte とし、さらに yaw_damping効果として微分に持たせるか、
        #    もしくは heading_error を PIDに加える。
        #    ここでは、D項が heading_error と同じ働きをする（d(cte)/dt = v*sin(heading_error)）ため、
        #    純粋に steer_error = -cte に対するPIDだけで、D項をしっかり効かせることで完璧に安定することを示します。
        steer_error = -cte
        steer_cmd = lat_pid.step(steer_error)
        
        # 状態更新
        accel, steer, jerk = vehicle.update(accel_cmd, steer_cmd)
        
        # 記録
        vehicle.record(t, accel, steer, jerk, cte, heading_error)
        
    return vehicle

def main():
    dt = 0.05
    duration = 18.0
    target_speed = 10.0 # m/s (時速 36 km/h)
    
    # 経路データの生成 (S字カーブ / 正弦波道路)
    path_x = np.linspace(0.0, 160.0, 1600)
    # y = 4.0 * sin(0.04 * x)
    path_y = 4.0 * np.sin(0.04 * path_x)
    
    # ウェイポイントでのTangent（接線方向の角度yaw）を計算
    path_yaw = np.zeros_like(path_x)
    for i in range(len(path_x) - 1):
        path_yaw[i] = math.atan2(path_y[i+1] - path_y[i], path_x[i+1] - path_x[i])
    path_yaw[-1] = path_yaw[-2]
    
    # --- 3種類のPIDゲインで走行テスト ---
    # 1. Tuned (適切に調整されたスムーズな制御)
    #   D項をしっかり効かせることで、ヨー角のふらつきを効果的にダンピングする
    lon_gains_tuned = (1.5, 0.05, 0.1)
    lat_gains_tuned = (0.22, 0.005, 0.35)
    
    # 2. Oscillating (Pゲインが高すぎてDが不足、激しくガタつく・蛇行する)
    lon_gains_high_p = (5.0, 0.0, 0.0)
    lat_gains_high_p = (0.6, 0.0, 0.02)
    
    # 3. Sluggish (ゲインが低すぎて曲がりきれず、カーブで大回り/インカットする)
    lon_gains_low_p = (0.3, 0.0, 0.0)
    lat_gains_low_p = (0.05, 0.0, 0.0)
    
    print("Running simulation for Tuned PID (Smooth)...")
    v_tuned = run_2d_simulation(path_x, path_y, path_yaw, target_speed, lon_gains_tuned, lat_gains_tuned, duration, dt)
    
    print("Running simulation for Oscillating PID (High P)...")
    v_osc = run_2d_simulation(path_x, path_y, path_yaw, target_speed, lon_gains_high_p, lat_gains_high_p, duration, dt)
    
    print("Running simulation for Sluggish PID (Low P)...")
    v_slug = run_2d_simulation(path_x, path_y, path_yaw, target_speed, lon_gains_low_p, lat_gains_low_p, duration, dt)
    
    # --- 静的プロット（評価グラフ）の描画と保存 ---
    print("Saving static tuning result plot...")
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    
    # (1) 2次元軌跡 (Trajectory)
    axs[0, 0].plot(path_x, path_y, 'k--', label='Reference Path', alpha=0.7)
    axs[0, 0].plot(v_tuned.history['x'], v_tuned.history['y'], '#00b894', label='Tuned (Smooth)', linewidth=2)
    axs[0, 0].plot(v_osc.history['x'], v_osc.history['y'], '#d63031', label='High P (Oscillating)', linewidth=1.5)
    axs[0, 0].plot(v_slug.history['x'], v_slug.history['y'], '#fdcb6e', label='Low P (Sluggish)', linewidth=1.5)
    axs[0, 0].set_title("2D Trajectory Tracking")
    axs[0, 0].set_xlabel("X Position (m)")
    axs[0, 0].set_ylabel("Y Position (m)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].axis('equal')
    
    # (2) 速度の追従性 (Speed Tracking)
    axs[0, 1].axhline(target_speed, color='k', linestyle='--', label='Target Speed (10m/s)', alpha=0.7)
    axs[0, 1].plot(v_tuned.history['t'], v_tuned.history['v'], '#00b894', label='Tuned', linewidth=2)
    axs[0, 1].plot(v_osc.history['t'], v_osc.history['v'], '#d63031', label='High P', linewidth=1.5)
    axs[0, 1].plot(v_slug.history['t'], v_slug.history['v'], '#fdcb6e', label='Low P', linewidth=1.5)
    axs[0, 1].set_title("Speed Tracking Profile")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Speed (m/s)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # (3) 横方向誤差 (Cross Track Error)
    axs[1, 0].axhline(0.0, color='k', linestyle='--', alpha=0.7)
    axs[1, 0].plot(v_tuned.history['t'], v_tuned.history['cte'], '#00b894', label='Tuned', linewidth=2)
    axs[1, 0].plot(v_osc.history['t'], v_osc.history['cte'], '#d63031', label='High P', linewidth=1.5)
    axs[1, 0].plot(v_slug.history['t'], v_slug.history['cte'], '#fdcb6e', label='Low P', linewidth=1.5)
    axs[1, 0].set_title("Cross-Track Error (CTE)")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Error (m)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # (4) ステアリング角 (Steering Command)
    axs[1, 1].plot(v_tuned.history['t'], np.degrees(v_tuned.history['steer']), '#00b894', label='Tuned', linewidth=2)
    axs[1, 1].plot(v_osc.history['t'], np.degrees(v_osc.history['steer']), '#d63031', label='High P', linewidth=1.5)
    axs[1, 1].plot(v_slug.history['t'], np.degrees(v_slug.history['steer']), '#fdcb6e', label='Low P', linewidth=1.5)
    axs[1, 1].set_title("Steering Angle Command")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Angle (deg)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # (5) 方位誤差 (Heading Error)
    axs[2, 0].axhline(0.0, color='k', linestyle='--', alpha=0.7)
    axs[2, 0].plot(v_tuned.history['t'], np.degrees(v_tuned.history['heading_error']), '#00b894', label='Tuned', linewidth=2)
    axs[2, 0].plot(v_osc.history['t'], np.degrees(v_osc.history['heading_error']), '#d63031', label='High P', linewidth=1.5)
    axs[2, 0].plot(v_slug.history['t'], np.degrees(v_slug.history['heading_error']), '#fdcb6e', label='Low P', linewidth=1.5)
    axs[2, 0].set_title("Heading Angle Error")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Error (deg)")
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    # (6) ジャーク (加加速度 - 乗り心地指標)
    axs[2, 1].plot(v_tuned.history['t'], v_tuned.history['jerk'], '#00b894', label='Tuned (Comfortable)', linewidth=2)
    axs[2, 1].plot(v_osc.history['t'], v_osc.history['jerk'], '#d63031', label='High P (Jumpy)', linewidth=1)
    # Sluggishはあまりに大人しいので描画
    axs[2, 1].plot(v_slug.history['t'], v_slug.history['jerk'], '#fdcb6e', label='Low P (Dull)', linewidth=1)
    axs[2, 1].set_ylim(-30, 30) # 見やすさのために範囲を固定
    axs[2, 1].set_title("Longitudinal Jerk (Comfort Index)")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Jerk (m/s^3)")
    axs[2, 1].legend()
    axs[2, 1].grid(True)
    
    plt.tight_layout()
    output_png = os.path.join(os.path.dirname(__file__), "pid_tuning_results.png")
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Static plot saved as: {output_png}")
    
    # --- アニメーション可視化 (GIF作成) ---
    print("Generating animated GIF simulation...")
    # 全ての位置ヒストリの長さを揃えるか、最小値に切り揃える
    min_steps = min(len(v_tuned.history['x']), len(v_osc.history['x']), len(v_slug.history['x']))
    
    # アニメーション用フィギュア設定
    fig_anim, ax_anim = plt.subplots(figsize=(12, 6))
    ax_anim.plot(path_x, path_y, 'k--', label='Reference Lane Center', alpha=0.5)
    
    # 各車の描画オブジェクトのプレースホルダー
    trail_tuned, = ax_anim.plot([], [], '#00b894', label='Tuned (Smooth)', linewidth=2)
    trail_osc, = ax_anim.plot([], [], '#d63031', label='High P (Oscillating)', linewidth=1)
    trail_slug, = ax_anim.plot([], [], '#fdcb6e', label='Low P (Sluggish)', linewidth=1.5)
    
    car_tuned = ax_anim.scatter([], [], color='#00b894', edgecolors='black', s=120, zorder=5, marker='s')
    car_osc = ax_anim.scatter([], [], color='#d63031', edgecolors='black', s=120, zorder=5, marker='s')
    car_slug = ax_anim.scatter([], [], color='#fdcb6e', edgecolors='black', s=120, zorder=5, marker='s')
    
    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, fontsize=11, fontweight='bold')
    info_text = ax_anim.text(0.02, 0.70, '', transform=ax_anim.transAxes, fontsize=10, 
                             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    ax_anim.set_xlim(-10, 170)
    ax_anim.set_ylim(-8, 8)
    ax_anim.set_xlabel("X Position (meters)")
    ax_anim.set_ylabel("Y Position (meters)")
    ax_anim.set_title("Autonomous Steering Control: PID Tuning Comparison", fontsize=12, fontweight='bold')
    ax_anim.legend(loc='lower right')
    ax_anim.grid(True)
    
    # ダウンサンプリングしてアニメーションの作成時間を削減 (1ステップ0.05秒 => 3ステップごとにプロット: 60ms/frame)
    step_interval = 3
    frames_indices = list(range(0, min_steps, step_interval))
    
    def init():
        trail_tuned.set_data([], [])
        trail_osc.set_data([], [])
        trail_slug.set_data([], [])
        car_tuned.set_offsets(np.empty((0, 2)))
        car_osc.set_offsets(np.empty((0, 2)))
        car_slug.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        info_text.set_text('')
        return trail_tuned, trail_osc, trail_slug, car_tuned, car_osc, car_slug, time_text, info_text

    def animate(i):
        idx = frames_indices[i]
        
        # 軌跡のアップデート
        trail_tuned.set_data(v_tuned.history['x'][:idx], v_tuned.history['y'][:idx])
        trail_osc.set_data(v_osc.history['x'][:idx], v_osc.history['y'][:idx])
        trail_slug.set_data(v_slug.history['x'][:idx], v_slug.history['y'][:idx])
        
        # 現在の車の位置
        car_tuned.set_offsets([[v_tuned.history['x'][idx], v_tuned.history['y'][idx]]])
        car_osc.set_offsets([[v_osc.history['x'][idx], v_osc.history['y'][idx]]])
        car_slug.set_offsets([[v_slug.history['x'][idx], v_slug.history['y'][idx]]])
        
        # テキストの更新
        t = v_tuned.history['t'][idx]
        time_text.set_text(f"Simulation Time: {t:.2f}s")
        
        info = (
            f"Tuned PID (Smooth):\n"
            f"  Speed: {v_tuned.history['v'][idx]:.1f} m/s | CTE: {v_tuned.history['cte'][idx]:.2f} m\n"
            f"High P (Oscillating):\n"
            f"  Speed: {v_osc.history['v'][idx]:.1f} m/s | CTE: {v_osc.history['cte'][idx]:.2f} m\n"
            f"Low P (Sluggish):\n"
            f"  Speed: {v_slug.history['v'][idx]:.1f} m/s | CTE: {v_slug.history['cte'][idx]:.2f} m"
        )
        info_text.set_text(info)
        
        return trail_tuned, trail_osc, trail_slug, car_tuned, car_osc, car_slug, time_text, info_text

    ani = animation.FuncAnimation(fig_anim, animate, init_func=init,
                                  frames=len(frames_indices), interval=100, blit=True)
    
    output_gif = os.path.join(os.path.dirname(__file__), "pid_simulation.gif")
    writer = animation.PillowWriter(fps=10)
    ani.save(output_gif, writer=writer)
    plt.close(fig_anim)
    print(f"Animated GIF saved as: {output_gif}")

if __name__ == "__main__":
    main()
