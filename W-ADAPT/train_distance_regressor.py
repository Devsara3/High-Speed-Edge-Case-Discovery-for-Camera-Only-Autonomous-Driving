import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PyTorchのインポート
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 距離推定AIモデル (多層パーセプトロン MLP)
if TORCH_AVAILABLE:
    class DistanceRegressor(nn.Module):
        def __init__(self):
            super(DistanceRegressor, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(8, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 1)
            )
            
        def forward(self, x):
            return self.fc(x)
else:
    DistanceRegressor = None

def collect_training_data(demo_mode=True, host='localhost', port=2000, target_samples=8000):
    """
    シミュレータ(実機/モック)をシーケンス走行させ、YOLO3Dの枠情報と本物距離のペアを収集します。
    """
    print(f"\n===== Start Collecting Distance Training Data (Target: {target_samples} samples) =====")
    from run_camera_only_experiment import CameraOnlyExperiment
    
    experiment = CameraOnlyExperiment(demo_mode=demo_mode, host=host, port=port)
    
    # 複数周回してデータを十分に溜める
    loop_count = 0
    try:
        # 天候は標準的で視界が良い状態に設定してベースの幾何関係を綺麗に学習させる
        experiment.set_weather_params(sun_altitude=90.0, precipitation=0.0, fog_density=0.0)
        
        while len(experiment.training_data) < target_samples:
            loop_count += 1
            print(f"\n--- Data Collection Loop {loop_count} (Current Samples: {len(experiment.training_data)}) ---")
            
            # シーケンスを1周走らせる
            experiment.run_sequence()
            
            # 无限ループ防止安全弁（mock/real共用）
            max_loops = 5 if demo_mode else 200
            if loop_count >= max_loops:
                print(f"[INFO] Reached max loop limit ({max_loops}) for safety.")
                break
                
    finally:
        csv_path = "results/distance_training_data.csv"
        experiment.export_training_data(csv_path)
        experiment.shutdown()
        
    print(f"[FINISHED] Collected total {len(experiment.training_data)} samples.")

def train_model(epochs=150, batch_size=128, lr=0.001):
    """
    収集されたCSVデータを用いてPyTorchモデルを学習させます。
    """
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is not installed. Cannot train the model.")
        return
        
    csv_path = "results/distance_training_data.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Training data file '{csv_path}' not found! Run with '--collect' first.")
        return
        
    # 1. データの読み込み
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} samples from {csv_path}")
    
    # 特徴量とターゲットの抽出
    features_cols = [
        'is_pedestrian', 'is_car', 'is_construction_signal', 'is_traffic_light',
        'bbox_y_bottom', 'bbox_height', 'bbox_width', 'ego_speed'
    ]
    
    X = df[features_cols].values.astype(np.float32)
    y = df['z_gt'].values.astype(np.float32).reshape(-1, 1)
    
    # データのシャッフルと分割 (Train 80% / Val 20%)
    indices = np.arange(len(df))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(len(df) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
    
    # DataLoaderの構築
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. モデル、損失関数、オプティマイザの定義
    model = DistanceRegressor()
    criterion = nn.HuberLoss(delta=1.0) # 外れ値に頑健なHuber Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # 3. 学習ループ
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 8 # 8エポック連続で改善しなければストップ
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # 訓練フェーズ
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss /= len(X_train)
        train_losses.append(epoch_train_loss)
        
        # 検証フェーズ
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item() * batch_X.size(0)
                
        epoch_val_loss /= len(X_val)
        val_losses.append(epoch_val_loss)
        
        # 学習率スケジューラーの適用
        scheduler.step(epoch_val_loss)
        
        # 最良モデルの保存と早期終了判定
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")
            
        if patience_counter >= early_stopping_patience:
            print(f"--> Early stopping triggered at epoch {epoch}. Restoring best model state.")
            break
            
    elapsed = time.time() - start_time
    print(f"\n[INFO] Training finished in {elapsed:.2f} seconds. Best Val Loss (Huber): {best_val_loss:.4f}")
    
    # 4. モデル重みのエクスポート
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, "distance_regressor.pth")
    print("[SUCCESS] DistanceRegressor model saved to 'distance_regressor.pth'")
    
    # 5. 損失推移の描画と保存
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Huber Loss', color='blue', linewidth=1.5)
    plt.plot(val_losses, label='Val Huber Loss', color='orange', linestyle='--', linewidth=1.5)
    plt.title("MLP Distance Regressor Training Progress", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Huber Loss", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("results/distance_training_loss.png", dpi=150)
    plt.close()
    print("[SUCCESS] Loss progress plot saved to 'results/distance_training_loss.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distance Regressor training pipeline.")
    parser.add_argument('--collect', action='store_true', help="Collect training data from sequence simulation")
    parser.add_argument('--train', action='store_true', help="Train MLP model using PyTorch")
    parser.add_argument('--demo', action='store_true', default=True, help="Data collection: run in mock/demo mode (default: True)")
    parser.add_argument('--nodemo', dest='demo', action='store_false', help="Data collection: run in live CARLA mode")
    parser.add_argument('--samples', type=int, default=8000, help="Target training samples to collect (default: 8000)")
    parser.add_argument('--epochs', type=int, default=150, help="Training epochs (default: 150)")
    parser.add_argument('--batch-size', type=int, default=128, help="Training batch size (default: 128)")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument('--host', type=str, default='localhost', help="CARLA host (default: localhost)")
    parser.add_argument('--port', type=int, default=2000, help="CARLA port (default: 2000)")
    args = parser.parse_args()
    
    if not args.collect and not args.train:
        parser.print_help()
        sys.exit(0)
        
    if args.collect:
        collect_training_data(demo_mode=args.demo, host=args.host, port=args.port, target_samples=args.samples)
        
    if args.train:
        train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
