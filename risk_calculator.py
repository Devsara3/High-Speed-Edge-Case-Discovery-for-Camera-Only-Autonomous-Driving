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
            'pedestrian': 1.8,
            'truck': 1.2,
            'car': 1.0,
            'bicycle': 1.3,
            'construction_signal': 1.5,
            'traffic_light': 2.0,
            'unknown': 1.0
        }

    def calculate_risk(self, ego_pos, ego_vel, target_pos, target_vel, target_class, yolo_z_distance):
        """
        単一障害物に対する知覚リスク R_perceived を計算します。
        
        :param ego_pos: 自車の位置ベクトル [x, y, z] (Ground Truth)
        :param ego_vel: 自車の速度ベクトル [vx, vy, vz] (Ground Truth)
        :param target_pos: 相手の位置ベクトル [x, y, z] (Ground Truth)
        :param target_vel: 相手の速度ベクトル [vx, vy, vz] (Ground Truth)
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
        alpha = np.exp(self.kappa * max(0, approach_speed))

        # 4. beta: 横方向の減衰因子（外積による衝突コース判定）
        ego_speed = np.linalg.norm(ego_vel)
        if ego_speed > 1e-3:
            ego_dir = ego_vel / ego_speed
            cross_prod = np.linalg.norm(np.cross(ego_dir, rel_pos_dir))
            sin_theta_sq = cross_prod ** 2
            beta = np.exp(-2.0 * sin_theta_sq)
        else:
            beta = 1.0

        # 5. mu: 車種係数
        mu = self.class_mu_table.get(target_class, 1.0)

        # 6. YOLOの主観的距離 (Prediction)
        r_hat = yolo_z_distance
        
        # 7. 最終的な知覚リスクの計算
        numerator = omega * mu * alpha * beta
        denominator = (r_hat ** 2) + self.epsilon
        
        r_perceived = self.K * (numerator / denominator) + self.C
        
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

    def calculate_multi_risk(self, ego_pos, ego_vel, gt_obstacles, yolo_detections):
        """
        複数のオブジェクトに対して、個別に知覚リスクと物理リスクを計算し、
        アクターごとの知覚ギャップ（r_gt - r_perceived）の最大値を選択して統合します。
        
        :param ego_pos: 自車位置 [x, y, z]
        :param ego_vel: 自車速度 [vx, vy, vz]
        :param gt_obstacles: 真値の障害物リスト [{'class':..., 'pos':..., 'vel':..., 'mu':...}, ...]
        :param yolo_detections: YOLOの検出結果リスト [{'class':..., 'z_distance':..., 'yolo3d_rel_pos':..., 'traffic_light_color':...}, ...]
        """
        max_r_perceived = 0.0
        max_r_gt = 0.0
        max_gap = -float('inf')
        worst_obstacle_name = 'unknown'
        worst_gt_dist = 0.0
        worst_yolo_dist = float('inf')
        per_obstacle_results = []
        
        remaining_detections = list(yolo_detections)
        
        for gt in gt_obstacles:
            gt_class = gt['class']
            gt_pos = np.array(gt['pos'], dtype=float)
            gt_vel = np.array(gt['vel'], dtype=float)
            
            # 自車とターゲットの距離 (GT)
            gt_dist = np.linalg.norm(gt_pos - np.array(ego_pos, dtype=float))
            
            # 同一クラスのYOLO検出から、距離が最も近いものを探す
            best_match = None
            best_match_idx = -1
            min_dist_diff = float('inf')
            
            for idx, det in enumerate(remaining_detections):
                det_class = det['class']
                if det_class in ['truck', 'bus', 'motorcycle', 'bicycle']:
                    det_class = 'car'
                
                check_class = gt_class
                if check_class in ['truck', 'bus', 'motorcycle', 'bicycle']:
                    check_class = 'car'
                
                if det_class == check_class:
                    diff = abs(det['z_distance'] - gt_dist)
                    if diff < min_dist_diff:
                        min_dist_diff = diff
                        best_match = det
                        best_match_idx = idx
                        
            # マッチングが成立した場合、検出距離を採用
            yolo_z = float('inf')
            detected_color = None
            if best_match is not None:
                yolo_z = best_match['z_distance']
                detected_color = best_match.get('traffic_light_color')
                remaining_detections.pop(best_match_idx)
                
            # 信号機ペナルティルール
            # 赤/黄信号で、検出漏れまたは青信号と誤認した場合は知覚距離無限大
            is_red_yellow_gt = gt_class == 'traffic_light' and gt.get('color') in ['red', 'yellow']
            if is_red_yellow_gt:
                if best_match is None or detected_color not in ['red', 'yellow']:
                    yolo_z = float('inf')
            
            # 各障害物のリスク算出
            r_perceived, _ = self.calculate_risk(
                ego_pos, ego_vel, gt_pos, gt_vel, gt_class, yolo_z
            )
            
            # 物理リスク (YOLOの主観距離として実際のGT距離を与える)
            r_gt, _ = self.calculate_risk(
                ego_pos, ego_vel, gt_pos, gt_vel, gt_class, gt_dist
            )
            
            # 危険度係数（mu）の適用
            r_perceived_scaled = r_perceived * (gt['mu'] / self.class_mu_table.get(gt_class, 1.0))
            r_gt_scaled = r_gt * (gt['mu'] / self.class_mu_table.get(gt_class, 1.0))
            
            # アクター個別のギャップ計算
            gap_i = r_gt_scaled - r_perceived_scaled
            
            if r_perceived_scaled > max_r_perceived:
                max_r_perceived = r_perceived_scaled
            if r_gt_scaled > max_r_gt:
                max_r_gt = r_gt_scaled
                
            # 最大ギャップ（最悪の見落としリスク）に基づいて最悪障害物を更新
            if gap_i > max_gap:
                max_gap = gap_i
                worst_obstacle_name = gt_class
                worst_gt_dist = gt_dist
                worst_yolo_dist = yolo_z
                
            per_obstacle_results.append({
                'class': gt_class,
                'gt_distance': gt_dist,
                'yolo_distance': yolo_z,
                'r_gt': r_gt_scaled,
                'r_perceived': r_perceived_scaled,
                'perception_gap': gap_i
            })
            
        # もし障害物が何もない場合の安全ガード
        if max_gap == -float('inf'):
            max_gap = 0.0
            
        return max_r_perceived, max_r_gt, max_gap, {
            'worst_obstacle': worst_obstacle_name,
            'worst_gt_distance': worst_gt_dist,
            'worst_yolo_distance': worst_yolo_dist,
            'details': per_obstacle_results
        }

class PerceivedRiskCalculator:
    """
    周辺の複数エージェントから受ける知覚リスクを、CARLA真値とYOLO3D予測値に基づいて算出するクラス。
    """
    def __init__(self, K=1.0, C=0.0, epsilon=1e-5, kappa=0.05, lambda_val=0.5):
        self.K = K
        self.C = C
        self.epsilon = epsilon
        self.kappa = kappa
        self.lambda_val = lambda_val

        # ② 相互作用の重み (omega) のデフォルト定数
        self.omega_bi = 2.0      # パターン1: 両者が接近
        self.omega_agent = 1.5   # パターン2: 相手だけが接近
        self.omega_ego = 1.0     # パターン3: 自車だけが接近
        self.omega_angr = 0.5    # パターン4: その他

        # ③ カテゴリ係数 (mu) のルックアップテーブル
        self.class_mu_table = {
            'truck': 1.5, 'bus': 1.5, 'trailer': 1.5,
            'car': 1.0, 'van': 1.0,
            'pedestrian': 1.2,
            'bicycle': 1.2, 'motorcycle': 1.2,
            'unknown': 1.0
        }

    def compute_frame_risk(self, ego_pos, ego_vel, targets_data):
        """
        周辺に存在するすべてのエージェントのリスクスコアを計算し、代表リスクスコア r_t を返却します。
        
        :param ego_pos: 自車の真の3D位置 (x, y, z)
        :param ego_vel: 自車の真の速度ベクトル (vx, vy, vz)
        :param targets_data: 各エージェントの予測および真値データ辞書のリスト
        :return: 統合代表リスクスコア (最大値), 詳細デバッグ情報
        """
        ego_pos = np.array(ego_pos, dtype=float)
        ego_vel = np.array(ego_vel, dtype=float)
        
        max_r_perceived = 0.0
        worst_obstacle = 'unknown'
        worst_gt_dist = 0.0
        worst_yolo_dist = float('inf')
        details = []

        for target in targets_data:
            true_pos = np.array(target['true_pos'], dtype=float)
            true_vel = np.array(target['true_vel'], dtype=float)
            class_type = target['class_type']
            yolo3d_rel_pos = target['yolo3d_rel_pos'] # [X_pred, Y_pred, Z_pred] 相対座標

            # 真の相対位置ベクトルの計算
            r_i = true_pos - ego_pos
            distance_gt = np.linalg.norm(r_i)
            
            if distance_gt < 1e-3:
                r_hat = np.array([1.0, 0.0, 0.0])
            else:
                r_hat = r_i / distance_gt

            # ① 分母：YOLO3Dによる主観的距離の計算
            if yolo3d_rel_pos is None:
                # 見落としている場合は主観的距離無限大とし、リスクは0とする
                r_perceived = 0.0
                yolo_dist = float('inf')
            else:
                yolo_rel = np.array(yolo3d_rel_pos, dtype=float)
                # 相対座標から直線距離の2乗を計算
                r_hat_sq = np.sum(yolo_rel ** 2)
                yolo_dist = np.sqrt(r_hat_sq)

                # ② 相互作用の重み (omega) の判定
                d_ego_to_target = np.dot(ego_vel, r_i)
                d_target_to_ego = np.dot(true_vel, -r_i)

                if d_ego_to_target > 0 and d_target_to_ego > 0:
                    omega = self.omega_bi
                elif d_ego_to_target <= 0 and d_target_to_ego > 0:
                    omega = self.omega_agent
                elif d_ego_to_target > 0 and d_target_to_ego <= 0:
                    omega = self.omega_ego
                else:
                    omega = self.omega_angr

                # ③ カテゴリ係数 (mu)
                mu = self.class_mu_table.get(class_type, 1.0)

                # ④ 縦方向（接近速度）の増幅因子 (alpha)
                v_rel = ego_vel - true_vel
                s_i = max(0.0, np.dot(v_rel, r_hat))
                alpha = np.exp(self.kappa * s_i)

                # ⑤ 横方向（進路逸れ）の減衰因子 (beta)
                v_rel_norm_sq = np.sum(v_rel ** 2)
                if v_rel_norm_sq > 1e-6:
                    cross_prod = np.cross(v_rel, r_hat)
                    cross_prod_norm_sq = np.sum(cross_prod ** 2)
                    sin2_theta = cross_prod_norm_sq / (v_rel_norm_sq + self.epsilon)
                else:
                    sin2_theta = 0.0
                
                beta = np.exp(-self.lambda_val * sin2_theta)

                # 知覚リスク R_perceived^t の計算
                numerator = omega * mu * alpha * beta
                denominator = r_hat_sq + self.epsilon
                r_perceived = self.K * (numerator / denominator) + self.C

            # 最も危険度が高いアクターの更新
            if r_perceived > max_r_perceived:
                max_r_perceived = r_perceived
                worst_obstacle = class_type
                worst_gt_dist = distance_gt
                worst_yolo_dist = yolo_dist

            details.append({
                'class': class_type,
                'gt_distance': distance_gt,
                'yolo_distance': yolo_dist,
                'r_perceived': r_perceived
            })

        debug_info = {
            'worst_obstacle': worst_obstacle,
            'worst_gt_distance': worst_gt_dist,
            'worst_yolo_distance': worst_yolo_dist,
            'details': details
        }

        return max_r_perceived, debug_info

if __name__ == '__main__':
    # 1. 既存の RiskCalculator のテスト
    print("Testing existing RiskCalculator...")
    calc = RiskCalculator()
    
    gt_obs = [
        {'class': 'pedestrian', 'pos': [10.0, 0, 0], 'vel': [0, 0, 0], 'mu': 1.8},
        {'class': 'car', 'pos': [25.0, 0, 0], 'vel': [10.0, 0, 0], 'mu': 1.0},
        {'class': 'traffic_light', 'pos': [20.0, 0, 0], 'vel': [0, 0, 0], 'color': 'red', 'mu': 2.0}
    ]
    
    detections = [
        {'class': 'pedestrian', 'z_distance': 10.0, 'traffic_light_color': None},
        {'class': 'car', 'z_distance': 25.0, 'traffic_light_color': None},
        {'class': 'traffic_light', 'z_distance': 20.0, 'traffic_light_color': 'green'}
    ]
    
    r_perc, r_gt, gap, info = calc.calculate_multi_risk(
        ego_pos=[0, 0, 0], ego_vel=[13.8, 0, 0],
        gt_obstacles=gt_obs, yolo_detections=detections
    )
    print(f"Multi risk calculation with Red Light Misclassification:")
    print(f"  Perceived Risk: {r_perc:.2f}, GT Risk: {r_gt:.2f}, Gap Score: {gap:.2f}")
    print(f"  Worst Obstacle: {info['worst_obstacle']}")

    # 2. 新しい PerceivedRiskCalculator のテスト
    print("\nTesting new PerceivedRiskCalculator...")
    new_calc = PerceivedRiskCalculator()
    
    targets_data = [
        {
            'true_pos': (10.0, 0.0, 0.0),
            'true_vel': (0.0, 0.0, 0.0),
            'class_type': 'pedestrian',
            'yolo3d_rel_pos': (10.0, 0.0, 0.0)
        },
        {
            'true_pos': (25.0, 0.0, 0.0),
            'true_vel': (-10.0, 0.0, 0.0), # 接近中
            'class_type': 'car',
            'yolo3d_rel_pos': (30.0, 0.0, 0.0) # 距離過大評価
        }
    ]
    
    r_t, dbg = new_calc.compute_frame_risk(
        ego_pos=(0.0, 0.0, 0.0),
        ego_vel=(13.8, 0.0, 0.0),
        targets_data=targets_data
    )
    print(f"Frame Risk (r_t): {r_t:.4f}")
    print(f"Worst Obstacle: {dbg['worst_obstacle']}")
    print(f"Worst GT Distance: {dbg['worst_gt_distance']:.2f}m")
    print(f"Worst YOLO Distance: {dbg['worst_yolo_distance']:.2f}m")
