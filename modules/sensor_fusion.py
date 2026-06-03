import joblib
import numpy as np
import os

class SensorFusionModule:
    def __init__(self, model_path='models/fusion_meta_learner.pkl'):
        # 事前に学習した重み予測モデルをロード
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            print("Warning: 学習済みモデルが見つかりません。デフォルトのルールベースを使用します。")
            self.model = None

    def fuse(self, precipitation, fog, sun_altitude, d_lidar, d_stereo, d_camera, d_ai, d_hsi):
        """
        現在の環境情報と各センサ値から、最適なフュージョン距離を計算して返す
        """
        if self.model:
            # 入力ベクトルを作成
            X = np.array([[precipitation, fog, sun_altitude, d_camera, d_lidar]])
            # モデルから重みを予測
            weights = self.model.predict(X)[0]
            
            # ソフトマックス的な後処理（念のための制約保証）
            weights = np.clip(weights, 0, 1)
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights /= weights_sum
            else:
                weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            # フォールバック用のルールベース
            if fog > 70:
                weights = [0.6, 0.1, 0.1, 0.1, 0.1]
            elif precipitation > 70:
                weights = [0.1, 0.1, 0.1, 0.3, 0.4]
            else:
                weights = [0.2, 0.3, 0.3, 0.1, 0.1]
            
        w_lidar, w_stereo, w_camera, w_ai, w_hsi = weights
        
        # 値が inf の場合の安全処理
        d_lidar_val = d_lidar if not np.isinf(d_lidar) else 100.0
        d_stereo_val = d_stereo if not np.isinf(d_stereo) else 100.0
        d_camera_val = d_camera if not np.isinf(d_camera) else 100.0
        d_ai_val = d_ai if not np.isinf(d_ai) else 100.0
        d_hsi_val = d_hsi if not np.isinf(d_hsi) else 100.0
        
        # フュージョン距離の算出
        d_fusion = (w_lidar * d_lidar_val) + (w_stereo * d_stereo_val) + (w_camera * d_camera_val) + (w_ai * d_ai_val) + (w_hsi * d_hsi_val)
        
        return d_fusion, weights
