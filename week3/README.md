# CARLA 接続・実行ガイド（開発メンバー向け）

このディレクトリに含まれる `carla_integration_demo.py` および各種モジュールを、実際にCARLAシミュレータに接続して実行するための手順書です。
セットアップを担当される方は、以下の手順に従って環境構築と実行を行ってください。

---

## 1. 必要な環境の準備 (Prerequisites)

CARLAのPython APIと、AI推論に必要なライブラリをインストールします。

```bash
# 1. 必須ライブラリのインストール
pip install numpy opencv-python

# 2. PyTorchのインストール (DeepLab, MiDaS等のAI推論用)
# ※GPU環境がある場合は、CUDA対応版をインストールしてください
pip install torch torchvision

# 3. CARLA Python APIのインストール
# ※お使いのCARLAサーバーのバージョン（例: 0.9.15）と完全に一致させる必要があります
pip install carla==0.9.15
```
*(※もし `pip install carla` でエラーが出る場合は、CARLA公式フォルダ内の `PythonAPI/carla/dist/` にある `.egg` または `.whl` ファイルを直接インストールしてください)*

---

## 2. CARLAサーバーの起動

Pythonスクリプトを実行する前に、必ず**CARLAサーバー（シミュレータ本体）を起動**しておく必要があります。

* **Windowsの場合**: コマンドプロンプト等でCARLAのフォルダを開き、実行します。
  ```cmd
  CarlaUE4.exe
  ```
  *(※PCのスペックが低く動作が重い場合は `CarlaUE4.exe -quality-level=Low` で起動すると軽くなります)*

シミュレータの画面が開き、街の風景が表示されればサーバーの準備は完了です。

---

## 3. 統合スクリプトの実行と接続

サーバーが起動したら、別のターミナル（コマンドプロンプト）を開き、本ディレクトリ内のメインスクリプトを実行します。

```bash
python carla_integration_demo.py
```

### 接続成功時に起こること
1. **`Connecting to CARLA Server...`**: デフォルトのポート `2000` 番を使ってサーバーと通信を確立します。
2. **AIモデルのロード**: MiDaSなどの初期化が行われます（初回のみダウンロードが入る場合があります）。
3. **車両のスポーン**: マップ上に Tesla Model 3 が出現し、ルーフにカメラが設置されます。
4. **自動運転ループ開始**: 車が自動で走り出し、別ウィンドウ（OpenCV）でカメラ画像と深度マップ（Depth Map）がリアルタイム表示されます。

---

## 4. プログラム内の「接続処理」の解説（エンジニア向け）
他の方がコードをカスタマイズする際の重要ポイントです。`carla_integration_demo.py` の以下の部分に注目してください。

### ① サーバーへの接続とタイムアウト設定
```python
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
```
* CARLAはデフォルトで `localhost` の `2000` 番ポートを使用します。
* **重要**: `set_timeout(10.0)` は必須です。初回の接続時や重いマップの読み込み時はデフォルトの2秒ではタイムアウトして落ちるため、長めに設定しています。

### ② 同期モード（Synchronous Mode）の有効化
```python
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS
world.apply_settings(settings)
```
* AI推論（数ミリ秒〜数十ミリ秒かかる）とシミュレータを組み合わせる場合、非同期のままだと「AIが計算している間にシミュレータの時間が進みすぎて車が壁に激突する」現象が起きます。
* これを防ぐため、**「Pythonスクリプトが `world.tick()` を呼ぶまで、シミュレータの時間は1ミリ秒も進まない」** という同期モードをオンにしています。
* `0.05` はPID制御の計算間隔（dt）と完全に一致させています。

### ③ 終了時のクリーンアップ（超重要）
```python
finally:
    settings.synchronous_mode = False
    world.apply_settings(settings)
```
* プログラムを強制終了（Ctrl+C）した際、同期モードをオフ（False）に戻さずに終わると、**CARLAシミュレータ本体の時間が永久にフリーズ**してしまい、CARLA自体の再起動が必要になります。`finally` ブロックで必ずオフに戻す処理を入れています。
