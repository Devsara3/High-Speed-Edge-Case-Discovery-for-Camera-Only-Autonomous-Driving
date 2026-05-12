# CARLA Edge Case Search Pipeline

このプロジェクトは、自動運転AI（YOLOv8等）の認識精度が低下する「エッジケース（悪天候条件）」を、効率的な探索アルゴリズム（Optuna）を用いて自動的に発見するためのパイプラインです。

現在は、CARLAシミュレータの代わりに画像処理ベースのモック環境 (`carla_mock.py`) で動作確認が可能です。

## プロジェクト構成

- `optimizer.py`: メインの実行スクリプト。Optunaを用いて天候パラメータを最適化（認識率を最小化）します。
- `carla_mock.py`: 指定された天候条件に基づいて、ベース画像 (`base_image.png`) に雨や霧のエフェクトをかけます。
- `evaluator.py`: YOLOv8を用いて、画像内の物体検出数や信頼度を評価します。
- `visualizer.py`: 探索結果の履歴を可視化（グラフ化）します。
- `carla_real_template.py`: 本物のCARLAシミュレータに接続するためのコードテンプレートです。

## セットアップ

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2. 動作確認 (モック環境)

まずはモック環境でパイプラインを動かしてみます。

```bash
python optimizer.py
```

実行後、`results/` ディレクトリに「最も認識が難しかった画像（エッジケース）」と「最も認識しやすかった画像」が保存されます。

## CARLAシミュレータとの統合方法

本物のCARLA環境で使用するには、以下の手順を行ってください。

1. **CARLA Simulatorのインストール**: [CARLA公式サイト](https://carla.org/)からシミュレータをダウンロードし、起動します。
2. **接続クラスの実装**: `carla_real_template.py` を参考に、実際のカメラセンサーから画像を取得するクラスを作成します。
3. **環境の差し替え**: `optimizer.py` 内で `MockCarlaEnv` をインポートしている箇所を、作成した新クラスに変更します。

```python
# optimizer.py
# from carla_mock import MockCarlaEnv
from my_carla_env import RealCarlaEnv

# env = MockCarlaEnv("base_image.png")
env = RealCarlaEnv()
```

## 貢献・共有について

このリポジトリをGitHubにプッシュする際は、`base_image.png` を含めることで、他のユーザーがすぐに動作確認できるようになっています。CARLA本体はリポジトリに含めず、各自の環境でセットアップすることを前提としています。
