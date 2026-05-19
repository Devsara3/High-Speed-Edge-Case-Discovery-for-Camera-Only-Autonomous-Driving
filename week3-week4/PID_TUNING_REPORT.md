# CARLA 統合開発レポート：マルチセンサーシステムとPID走行制御（Week 3 - Week 4）

本ドキュメントは、Gitリポジトリをクローンした開発チームメンバー向けに、CARLAシミュレータ上で自動運転車両（Ego Vehicle）に各種センサーを配置し、静的障害物の回避環境下で安定した車線追従・速度制御（PID）を検証するための手順と技術的な詳細をまとめたレポートです。

本プロジェクトの `week3-week4` フォルダ内は、AI関連の重いライブラリ（PyTorch等）を使用せず、CARLAのネイティブセンサーおよび数学的なPID演算のみで動作するため、あらゆる開発PC環境で非常に軽量に動作します。

---

## 1. システム構成と各チュートリアルステップの対応

本システムは、配布された Colab チュートリアルの手順を統合し、以下のように実装しています。

- **Step 3 (Ego車両のスポーン)**: Tesla Model 3をマップ内の安全なスポーンポイントへ配置します。
- **Step 4 (マルチセンサーの搭載・設定)**:
  - **RGBカメラ & セマンティックセグメンテーションカメラ**: フロントガラス高さ（`x=1.5, z=1.7`）に重ねて設置し、同一の画角（FOV 90）で生映像と色分けラベル付き映像を並行して取得。
  - **LiDAR**: ルーフ中央（`z=2.5`）に配置し、50mの検知距離、32チャンネル、同期20 FPSで点群数を取得。
  - **Radar**: フロントバンパー前部（`x=2.0, z=0.5`）に配置し、前方100m以内の検知点数を追跡。
  - **IMU（慣性計測ユニット） & GNSS（GPS受信機）**: 車体中心およびルーフから加速度、ヨーレート（方位角）、経度・緯度・高度をリアルタイム計測。
- **Step 5 (静的障害物の配置)**: 
  - 自車（Ego）の進行方向ベクトルを取得し、前方12mに静的障害物（Tesla Model 3）を自動配置。自動運転中に障害物が目の前に存在する環境を模擬します。
- **Step 6 & 7 (センサー出力のパースと可視化ダッシュボード)**:
  - OpenCVを用いて、左側に「生カメラ映像＋PID制御 telemetry」、右側に「セマンティックカメラ映像＋全センサーデータ（LiDAR点数、Radar点数、IMU加速度/角速度/方位角、GPS緯度/経度）」を並べた**リアルタイムダッシュボード**を生成します。

---

## 2. センサーデータの解析・処理設計（`carla_sensor_manager.py`）

各センサーから得られる生データバッファ（bytes）を、Python側で利用しやすいようにパースしています。

1. **RGB画像 ＆ セマンティックセグメンテーション画像**:
   セマンティック側は `carla.ColorConverter.CityScapesPalette` を適用して色分けした状態で、RGB配列へと変換して取得します。
   ```python
   # セマンティックカメラの画像パース処理
   image.convert(carla.ColorConverter.CityScapesPalette)
   array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
   array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1] # RGB化
   ```
2. **LiDAR / Radar**:
   1Dバイトバッファを、NumPyを用いて $[x, y, z, \text{intensity}]$ (LiDAR) や $[\text{velocity}, \text{azimuth}, \text{altitude}, \text{depth}]$ (Radar) の2次元配列（行列）に整形します。
3. **IMU / GNSS (GPS)**:
   加速度（Vector3D）、角速度（Vector3D）、方位角（compass / ラジアン）、緯度・経度・高度のそれぞれの物理値を取得し、辞書構造に保存してメインループへ渡します。

---

## 3. PID制御とアンチワインドアップ（`carla_pid_controller.py`）

車両操舵には制限（ステアリング角 $-1.0$ 〜 $1.0$）があるため、偏差が持続した際に積分項が過剰に蓄積する**積分ワインドアップ現象**を防ぐ**アンチワインドアップ（クランプ処理）**を実装し、走行の急激なふらつきを抑えています。

```python
# 横方向のクランプ例
if self.lat_ki != 0.0:
    max_lat_int = 1.0 / self.lat_ki
    self.lat_integral = max(-max_lat_int, min(max_lat_int, self.lat_integral))
```

### パラメータの物理的な役割
- **P（比例ゲイン: $K_p$）**: 車線中心へ戻そうとする復元力。
- **D（微分ゲイン: $K_d$）**: 進路修正時の首振りを打ち消す「ヨー角速度のダンパー」として機能。ガタつき（高周波のふらつき）を抑えるために最も重要です。
- **I（積分ゲイン: $K_i$）**: 旋回中や傾斜地での定常偏差の解消。

---

## 4. クローンしたメンバーの実行・接続手順

CARLAおよび本リポジトリがPCにセットアップされている場合、以下の手順ですぐにダッシュボードを実行できます。

### 4-1. 依存ライブラリの確認
```bash
pip install numpy opencv-python matplotlib
pip install carla==0.9.15  # シミュレータのバージョンに合わせる
```

### 4-2. CARLAの起動（Low品質モード推奨）
```bash
CarlaUE4.exe -quality-level=Low
```

### 4-3. 統合デモの実行
```bash
# week3-week4ディレクトリへ移動
cd High-Speed-Edge-Case-Discovery-for-Camera-Only-Autonomous-Driving/week3-week4

# 実行
python carla_integration_demo.py
```
*(実行すると自動的に障害物が前方に配置され、各種センサーの値をオーバーレイ表示した2分割のダッシュボードウィンドウが表示されます)*

### 4-4. パラメータの外部指定チューニング
```bash
# 横G・ステアのふらつきを抑える推奨パラメータでの実行
python carla_integration_demo.py --kp-lat 0.25 --kd-lat 0.35 --kp-lon 1.5 --kd-lon 0.1
```

---

## 5. 自動ログ評価と可視化

走行中に **`Ctrl+C`** で終了すると、クリーンアップ後に以下のファイルが `week3-week4` フォルダ内に保存されます。

1. **`carla_pid_tuning_results.png` (走行データグラフ)**: 
   速度追従、クロス・トラック・エラー（CTE / 横ズレ量）、ステアリング角、ペダル入力の時系列変化。
2. **`carla_run_recording.avi` (走行ダッシュボード動画)**: 
   RGB映像、セグメンテーション映像、および全センサー情報がリアルタイム更新されるダッシュボードをそのまま録画保存した動画。
