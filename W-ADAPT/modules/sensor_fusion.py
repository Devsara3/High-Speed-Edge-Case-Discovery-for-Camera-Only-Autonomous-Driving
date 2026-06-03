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

    def fuse(self, precipitation, fog, d_ai, d_stereo, d_lidar):
        """
        現在の環境情報と各センサ値から、最適なフュージョン距離を計算して返す
        """
        if self.model:
            # 入力ベクトルを作成
            X = np.array([[precipitation, fog, d_stereo, d_lidar]])
            # モデルから重み[W_ai, W_stereo, W_lidar]を予測
            weights = self.model.predict(X)[0]
            
            # ソフトマックス的な後処理（念のための制約保証）
            weights = np.clip(weights, 0, 1)
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights /= weights_sum
            else:
                weights = [0.33, 0.33, 0.34]
        else:
            # フォールバック用のルールベース（プラン1の初期値）
            if fog > 70:
                weights = [0.1, 0.1, 0.8]
            elif precipitation > 70:
                weights = [0.7, 0.2, 0.1]
            else:
                weights = [0.1, 0.6, 0.3]
            
        w_ai, w_stereo, w_lidar = weights
        
        # d_ai, d_stereo, d_lidar が inf の場合の安全処理
        d_ai_val = d_ai if not np.isinf(d_ai) else 100.0
        d_stereo_val = d_stereo if not np.isinf(d_stereo) else 100.0
        d_lidar_val = d_lidar if not np.isinf(d_lidar) else 100.0
        
        # フュージョン距離の算出
        d_fusion = (w_ai * d_ai_val) + (w_stereo * d_stereo_val) + (w_lidar * d_lidar_val)
        
        return d_fusion, weights
