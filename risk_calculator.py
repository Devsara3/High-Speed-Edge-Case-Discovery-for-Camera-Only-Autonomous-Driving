import numpy as np

class RiskCalculator:
    """
    CARLAの真値（Ground Truth）と、YOLOの主観的認識値（Prediction）をハイブリッドで組み合わせ、
    『知覚リスク（Perceived Risk）』を計算するクラス。
    """
    def __init__(self, K=100.0, C=0.0, epsilon=0.1, kappa=0.05):
        self.K = K
        self.C = C
        self.epsilon = epsilon
        self.kappa = kappa
        
        # 車種に応じた危険度係数（mu_i）のルックアップテーブル
        self.class_mu_table = {
            'pedestrian': 1.5,
            'truck': 1.2,
            'car': 1.0,
            'bicycle': 1.3,
            'unknown': 1.0
        }

    def calculate_risk(self, ego_pos, ego_vel, target_pos, target_vel, target_class, yolo_z_distance):
        """
        知覚リスク R_perceived を計算します。
        
        :param ego_pos: 自車の位置ベクトル [x, y, z] (Ground Truth)
        :param ego_vel: 自車の速度ベクトル [vx, vy, vz] (Ground Truth)
        :param target_pos: 相手車の位置ベクトル [x, y, z] (Ground Truth)
        :param target_vel: 相手車の速度ベクトル [vx, vy, vz] (Ground Truth)
        :param target_class: 相手のクラス名 (Ground Truth)
        :param yolo_z_distance: YOLO3Dが推定した相手までの奥行き(Z)距離 (Prediction)
        :return: 計算されたリスクスコア
        """
        ego_pos = np.array(ego_pos, dtype=float)
        ego_vel = np.array(ego_vel, dtype=float)
        target_pos = np.array(target_pos, dtype=float)
        target_vel = np.array(target_vel, dtype=float)

        # 1. 相対ベクトルと相対速度の計算 (Ground Truthベース)
        rel_pos = target_pos - ego_pos
        distance_gt = np.linalg.norm(rel_pos)
        
        # 距離が0に近すぎる場合のエラー回避
        if distance_gt < 1e-3:
            rel_pos_dir = np.array([1.0, 0.0, 0.0])
        else:
            rel_pos_dir = rel_pos / distance_gt
            
        rel_vel = ego_vel - target_vel # 自車から見た相手の相対接近速度ベクトル

        # 2. omega: 相互作用の重み（内積による接近判定）
        # 相対速度が相手の方向に向かっているか（正なら接近、負なら離脱）
        approach_speed = np.dot(rel_vel, rel_pos_dir)
        
        if approach_speed > 0:
            omega = 1.0 # 接近中
        else:
            omega = 0.1 # 離脱中（リスク低）

        # 3. alpha: 接近速度による増幅因子（指数関数）
        # s_i^t はクロージングスピード (approach_speed)
        alpha = np.exp(self.kappa * max(0, approach_speed))

        # 4. beta: 横方向の減衰因子（外積による衝突コース判定）
        # 自車の進行方向と、相手の方向のズレを計算
        ego_speed = np.linalg.norm(ego_vel)
        if ego_speed > 1e-3:
            ego_dir = ego_vel / ego_speed
            # 外積の大きさ (sin(theta))
            cross_prod = np.linalg.norm(np.cross(ego_dir, rel_pos_dir))
            sin_theta_sq = cross_prod ** 2
            # 正面(sin=0)なら beta=1.0、真横(sin=1)なら betaが小さくなるようにする
            beta = np.exp(-2.0 * sin_theta_sq)
        else:
            beta = 1.0

        # 5. mu: 車種係数
        mu = self.class_mu_table.get(target_class, 1.0)

        # 6. YOLOの主観的距離 (Prediction)
        # 検出失敗時は yolo_z_distance が無限大として渡される想定
        r_hat = yolo_z_distance
        
        # 7. 最終的な知覚リスクの計算
        numerator = omega * mu * alpha * beta
        denominator = (r_hat ** 2) + self.epsilon
        
        r_perceived = self.K * (numerator / denominator) + self.C
        
        # デバッグ用情報
        debug_info = {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'mu': mu,
            'approach_speed': approach_speed,
            'r_hat_yolo': r_hat,
            'gt_distance': distance_gt,
            'numerator': numerator,
            'denominator': denominator
        }

        return r_perceived, debug_info

if __name__ == '__main__':
    # 単体テスト
    calc = RiskCalculator()
    
    # GT: 自車は50km/h(約13.8m/s)で前進、相手は15m先で停止中
    r, debug = calc.calculate_risk(
        ego_pos=[0, 0, 0], ego_vel=[13.8, 0, 0],
        target_pos=[15.0, 0, 0], target_vel=[0, 0, 0],
        target_class='car',
        yolo_z_distance=15.0 # YOLOが正しく認識している場合
    )
    print(f"Normal recognition risk: {r:.2f}")
    
    # GT: 全く同じ物理状況だが、悪天候でYOLOが見落とした場合（距離=無限大）
    r_miss, debug_miss = calc.calculate_risk(
        ego_pos=[0, 0, 0], ego_vel=[13.8, 0, 0],
        target_pos=[15.0, 0, 0], target_vel=[0, 0, 0],
        target_class='car',
        yolo_z_distance=float('inf') # YOLOが見落とし
    )
    print(f"Weather trap (Missed detection) risk: {r_miss:.2f}")
