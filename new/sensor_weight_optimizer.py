import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

class SensorWeightOptimizer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            self.df = None
            print(f"Warning: {csv_path} not found. Please run the data collection (Phase 2) first.")
        
    def optimize_weights_for_row(self, row):
        """1ステップ（1行）における理想の重みをScipyで逆算"""
        d_gt = row['dist_gt']
        d_ai = row['dist_ai']
        d_stereo = row['dist_camera'] # CSVでは dist_camera として保存されています
        d_lidar = row['dist_lidar']
        
        # 目的関数: 二乗誤差 (RMSEのベース)
        def loss_fn(weights):
            w_ai, w_stereo, w_lidar = weights
            d_fusion = (w_ai * d_ai) + (w_stereo * d_stereo) + (w_lidar * d_lidar)
            return (d_fusion - d_gt) ** 2
        
        # 制約条件: 重みの合計は1.0, 各重みは0~1の間
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0, 1), (0, 1), (0, 1)]
        
        # 初期値は均等
        init_weights = [0.33, 0.33, 0.33]
        res = minimize(loss_fn, init_weights, bounds=bounds, constraints=constraints)
        return res.x if res.success else init_weights

    def train_meta_learner(self):
        """全データから天候とセンサ値を入力、最適重みを出力するモデルを学習"""
        if self.df is None:
            print("No data available to train the Meta-Learner.")
            return

        print("理想的な重みを計算中...")
        # 欠損値やエラー値(inf)を除外
        valid_df = self.df.replace([np.inf, -np.inf], np.nan).dropna(subset=['dist_gt', 'dist_ai', 'dist_camera', 'dist_lidar'])
        
        if len(valid_df) == 0:
            print("Error: 有効なデータ行が存在しません。")
            return
            
        optimized_w = valid_df.apply(self.optimize_weights_for_row, axis=1)
        
        # 教師データの作成
        X = valid_df[['precipitation', 'fog', 'dist_camera', 'dist_lidar']].values
        y = np.array(optimized_w.tolist())
        
        # ランダムフォレスト回帰でモデル化
        print("Meta-Learnerを学習中...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # モデルの保存
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/fusion_meta_learner.pkl')
        print("Meta-Learnerモデルの保存が完了しました。(models/fusion_meta_learner.pkl)")

if __name__ == "__main__":
    optimizer = SensorWeightOptimizer("results/optuna_history_sequence_Random.csv")
    optimizer.train_meta_learner()
