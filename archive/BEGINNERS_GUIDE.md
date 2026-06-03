# このプロジェクトを動かすための完全ガイド

このガイドは、プログラミングやGitHubが初めての方でも、迷わずにこの「自動運転エッジケース探索パイプライン」を動かせるように１ステップずつ解説します。



## ステップ2：プロジェクトを自分のパソコンにコピーする（クローン）
GitHubにあるこのプログラムを、自分のパソコンにダウンロードします。

1.  パソコンで適当なフォルダ（デスクトップなど）を開き、右クリックして「ターミナルで開く」を選択します。
2.  以下の命令をコピーして貼り付け、エンターキーを押してください：
    ```bash
    git clone https://github.com/Devsara3/High-Speed-Edge-Case-Discovery-for-Camera-Only-Autonomous-Driving.git
    ```
3.  ダウンロードが終わったら、新しくできた `High-Speed-Edge-Case-Discovery-for-Camera-Only-Autonomous-Driving` フォルダの中に入ります。
    ```bash
    cd High-Speed-Edge-Case-Discovery-for-Camera-Only-Autonomous-Driving
    ```

---

## ステップ3：必要なライブラリを入れる
このプログラムを動かすには、いくつかの追加パーツ（ライブラリ）が必要です。

1.  ターミナルで以下の命令を入力します：
    ```bash
    pip install -r requirements.txt
    ```
    ※ これで、AI（YOLO）や最適化ツール（Optuna）が自動的にインストールされます。

---

## ステップ4：実行してみる！
まずは、シミュレーターがなくても動く「モック版（お試し版）」を動かしてみましょう。

1.  **Windowsの方**：
    フォルダの中にある `run_pipeline.bat` をダブルクリックしてください。
2.  **Mac/Linuxの方**：
    ターミナルで `bash run_pipeline.sh` と入力してください。

---

## ステップ5：結果を確認する
プログラムが終わると、`results` というフォルダに結果が保存されます。

- **`edge_case_worst_TPE.jpg`**: AIが一番「危ないのに気づけなかった」瞬間です。
- **`history_tpe.csv`**: 全ての実験結果が記録されたエクセル形式のファイルです。

---

## 次のステップ：CARLA（シミュレーター）で動かすには
慣れてきたら、本物の3Dシミュレーター「CARLA」をインストールして動かしてみましょう。
詳しい手順は [CARLA_INTEGRATION_GUIDE.md](./CARLA_INTEGRATION_GUIDE.md) に書いてあります。

---

