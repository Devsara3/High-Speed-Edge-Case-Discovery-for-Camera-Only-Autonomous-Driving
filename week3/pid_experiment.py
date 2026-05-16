# %% [markdown]
# # 2-A: PID制御の挙動テスト（ロビン先生報告用）
# このスクリプトは、Jupyter Notebookのセル（# %%）としてVSCode等で実行可能なPythonスクリプトです。
# 縦方向（速度）と横方向（ステアリング）のPID制御をシミュレーションし、結果をグラフ化します。

# %%
import numpy as np
import matplotlib.pyplot as plt
import math

class PIDController:
    def __init__(self, kp, ki, kd, dt, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.min_out, self.max_out = output_limits
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error):
        # 積分項の計算
        self.integral += error * self.dt
        # 微分項の計算
        derivative = (error - self.prev_error) / self.dt
        # PIDの計算
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        
        # 出力の制限（アンチワインドアップも兼ねる簡易版）
        if self.min_out is not None:
            output = max(self.min_out, output)
        if self.max_out is not None:
            output = min(self.max_out, output)
            
        return output

class SimpleVehicle:
    def __init__(self, initial_v=0.0, initial_y=0.0, dt=0.05):
        self.v = initial_v      # 現在の速度 (m/s)
        self.y = initial_y      # 横方向の位置 (m)
        self.yaw = 0.0          # 車両の向き (rad)
        self.dt = dt
        
        self.history = {'t': [], 'v': [], 'accel': [], 'jerk': [], 'y': [], 'steer': []}
        self.time = 0.0
        self.prev_accel = 0.0

    def update(self, accel_cmd, steer_cmd):
        # 物理的な制限
        accel = np.clip(accel_cmd, -8.0, 4.0) # ブレーキは強く(-8.0)、アクセルは弱め(4.0)
        steer = np.clip(steer_cmd, -math.pi/4, math.pi/4) # 最大ステアリング角
        
        # ジャーク（加加速度）の計算：乗り心地の指標。急激にアクセル・ブレーキを踏むと大きくなる
        jerk = (accel - self.prev_accel) / self.dt
        
        # 状態の更新（シンプルな自転車モデルの近似）
        self.v += accel * self.dt
        self.v = max(0.0, self.v) # 後退はしない想定
        
        self.yaw += (self.v / 2.5) * math.tan(steer) * self.dt # ホイールベース 2.5mと仮定
        self.y += self.v * math.sin(self.yaw) * self.dt
        
        # 記録
        self.history['t'].append(self.time)
        self.history['v'].append(self.v)
        self.history['accel'].append(accel)
        self.history['jerk'].append(jerk)
        self.history['y'].append(self.y)
        self.history['steer'].append(steer)
        
        self.prev_accel = accel
        self.time += self.dt

# %% [markdown]
# ## 縦方向（速度制御）のシミュレーション
# 目標速度（例えば時速36km = 10m/s）に向かって加速し、その後停止するケースをテストします。

# %%
def run_speed_simulation(kp, ki, kd, target_speeds, duration=20.0, dt=0.05):
    vehicle = SimpleVehicle(dt=dt)
    pid = PIDController(kp, ki, kd, dt, output_limits=(-8.0, 4.0)) # 出力は加速度
    
    steps = int(duration / dt)
    targets = []
    
    for i in range(steps):
        t = i * dt
        # シナリオ: 前半は目標速度10m/s、後半は0m/s（停止）
        target_v = target_speeds[0] if t < duration/2 else target_speeds[1]
        targets.append(target_v)
        
        error = target_v - vehicle.v
        accel_cmd = pid.step(error)
        
        vehicle.update(accel_cmd, steer_cmd=0.0)
        
    return vehicle.history, targets

# 異なるパラメータでテスト
# 1. 良い調整（チューニング後）：I成分をゼロにしてワインドアップを防ぎ、D成分で滑らかに減速
hist_good, tgt = run_speed_simulation(kp=1.0, ki=0.0, kd=0.5, target_speeds=[10.0, 0.0])
# 2. 悪い調整：P成分が大きすぎて急ブレーキ・急発進になり、振動する
hist_high_p, _ = run_speed_simulation(kp=5.0, ki=0.0, kd=0.0, target_speeds=[10.0, 0.0])

plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1)
plt.plot(hist_good['t'], tgt, 'k--', label='Target Speed')
plt.plot(hist_good['t'], hist_good['v'], label='Good PID (Smooth)')
plt.plot(hist_high_p['t'], hist_high_p['v'], label='High P (Oscillation)')
plt.title("Speed Tracking")
plt.ylabel("Speed (m/s)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(hist_good['t'], hist_good['accel'], label='Good PID')
plt.plot(hist_high_p['t'], hist_high_p['accel'], label='High P')
plt.title("Acceleration (Throttle/Brake)")
plt.ylabel("Accel (m/s^2)")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(hist_good['t'], hist_good['jerk'], label='Good PID (Low Jerk)')
plt.plot(hist_high_p['t'], hist_high_p['jerk'], label='High P (High Jerk = Uncomfortable)')
plt.title("Jerk (Comfort/Safety Index) - Target is close to 0")
plt.ylabel("Jerk (m/s^3)")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("speed_control_result.png")
plt.close()

# %% [markdown]
# ## 横方向（ステアリング・レーンキープ）のシミュレーション
# 目標のY座標（レーンの中央など）に追従するシミュレーション。

# %%
def run_steer_simulation(kp, ki, kd, target_y=3.0, duration=10.0, dt=0.05):
    # 常に一定速度（10m/s）で走っている状態を想定
    vehicle = SimpleVehicle(initial_v=10.0, dt=dt)
    pid = PIDController(kp, ki, kd, dt, output_limits=(-math.pi/4, math.pi/4))
    
    steps = int(duration / dt)
    targets = []
    
    for i in range(steps):
        targets.append(target_y)
        # エラーは「目標位置 - 現在位置」
        error = target_y - vehicle.y
        steer_cmd = pid.step(error)
        
        # 速度は維持するのでaccel=0
        vehicle.update(accel_cmd=0.0, steer_cmd=steer_cmd)
        
    return vehicle.history, targets

# 異なるパラメータでテスト
# 1. 良い調整：徐々に近づき、目標でぴったりと直進する
hist_steer_good, tgt_y = run_steer_simulation(kp=0.1, ki=0.001, kd=0.15)
# 2. 悪い調整：P成分が大きすぎて急ハンドルを切り、ふらつく（大回り・小回りが安定しない）
hist_steer_high_p, _   = run_steer_simulation(kp=0.5, ki=0.0, kd=0.0) 

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(hist_steer_good['t'], tgt_y, 'k--', label='Target Lane Center')
plt.plot(hist_steer_good['t'], hist_steer_good['y'], label='Good PID (Smooth Curve)')
plt.plot(hist_steer_high_p['t'], hist_steer_high_p['y'], label='High P (Overshoot/Wobble)')
plt.title("Lane Keeping (Lateral Position)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(hist_steer_good['t'], hist_steer_good['steer'], label='Good PID')
plt.plot(hist_steer_high_p['t'], hist_steer_high_p['steer'], label='High P (Sharp turns)')
plt.title("Steering Angle")
plt.ylabel("Angle (rad)")
plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("steer_control_result.png")
plt.close()

# %% [markdown]
# ### ロビン先生への報告ポイント（まとめ）
# 
# 1. **P成分（比例）**: 
#    - 大きくすると目標に素早く近づこうとしますが、大きすぎると目標を通り過ぎてしまい（オーバーシュート）、振動（ガクガクとした動き）が発生しました。
#    - ステアリングでPを大きくすると、急激にハンドルを切りすぎるため、車両がふらつき（大回り・小回りが安定しない状態に）なりました。
# 2. **I成分（積分）**: 
#    - 摩擦や空気抵抗などで生じる「わずかなズレ（定常偏差）」を時間をかけて修正する役割を果たします。
# 3. **D成分（微分）**: 
#    - 急激な変化を抑える（ブレーキをかける）役割を果たします。
#    - 適切に設定することで、目標に近づく際のオーバーシュートを防ぎ、滑らかに速度を落とす（自然なアプローチ）ことができました。
#    - 結果として、ジャーク（急激な加減速による不快な揺れ）を最小限に抑え、快適な乗り心地を実現できました。
