import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import os

# --- 設定 ---
artifact_dir = r"C:\Users\濱田　紗空\.gemini\antigravity\brain\9d20c65b-f6da-4236-a55a-68635dc6dd9e"
md_path = os.path.join(artifact_dir, "model_transparency_report.md")

# フォント設定（日本語が文字化けしないように）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'Noto Sans CJK JP', 'sans-serif']

# --- データロード ---
model = joblib.load('models/fusion_meta_learner.pkl')
df = pd.read_csv('results/experiment_log_20260604_230118.csv')
valid_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['dist_gt', 'dist_ai', 'dist_camera', 'dist_lidar'])

X = valid_df[['precipitation', 'fog', 'dist_camera', 'dist_lidar']].values
dist_gt = valid_df['dist_gt'].values
dist_ai = valid_df['dist_ai'].values
dist_camera = valid_df['dist_camera'].values
dist_lidar = valid_df['dist_lidar'].values

pred_weights = model.predict(X)
d_meta = (pred_weights[:, 0] * dist_ai) + (pred_weights[:, 1] * dist_camera) + (pred_weights[:, 2] * dist_lidar)
valid_df['d_meta'] = d_meta
valid_df['w_ai'] = pred_weights[:, 0]
valid_df['w_camera'] = pred_weights[:, 1]
valid_df['w_lidar'] = pred_weights[:, 2]

# --- 1. 特徴量重要度 (Feature Importances) ---
importances = model.feature_importances_
feature_names = ['Precipitation (降水)', 'Fog (霧)', 'Dist Camera (カメラ距離)', 'Dist LiDAR (LiDAR距離)']

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title("Feature Importance (どの要因が重み決定に重要か)")
plt.xlabel("Importance Score")
feat_imp_path = os.path.join(artifact_dir, "feature_importance.png").replace("\\", "/")
plt.savefig(feat_imp_path, bbox_inches='tight')
plt.close()

# --- 2. 条件別の誤差 (RMSE) 分析 ---
def get_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# 条件ごとのバケット作成
results = []
conditions = {
    "全体": valid_df,
    "濃霧 (Fog > 50)": valid_df[valid_df['fog'] > 50],
    "大雨 (Precipitation > 50)": valid_df[valid_df['precipitation'] > 50],
    "晴天 (Fog<=20 & Precip<=20)": valid_df[(valid_df['fog'] <= 20) & (valid_df['precipitation'] <= 20)],
    "遠距離 (GT Distance > 50m)": valid_df[valid_df['dist_gt'] > 50],
    "近距離 (GT Distance <= 20m)": valid_df[valid_df['dist_gt'] <= 20],
}

for name, subset in conditions.items():
    if len(subset) == 0:
        continue
    rmse_ai = get_rmse(subset['dist_gt'], subset['dist_ai'])
    rmse_cam = get_rmse(subset['dist_gt'], subset['dist_camera'])
    rmse_lidar = get_rmse(subset['dist_gt'], subset['dist_lidar'])
    rmse_meta = get_rmse(subset['dist_gt'], subset['d_meta'])
    
    results.append({
        "Condition": name,
        "Samples": len(subset),
        "AI": rmse_ai,
        "Camera": rmse_cam,
        "LiDAR": rmse_lidar,
        "Meta-Learner": rmse_meta
    })

res_df = pd.DataFrame(results)

# --- 3. Markdownレポートの作成 ---
md_content = """# Meta-Learner 統計的透明性レポート

ブラックボックスになりがちな機械学習モデル（ランダムフォレスト）について、**「モデルが何を基準に判断しているか」「どのような条件下でどの程度のエラーが出ているか」**を統計的に分析し、透明性を確保するためのレポートです。

> [!NOTE]
> ランダムフォレストは決定木の集合体であり、どの特徴量（入力変数）が分岐に最も寄与したかを定量的に算出することが可能です。

## 1. 特徴量重要度（モデルの判断基準）

モデルがセンサーの重みを決定する際、どの入力情報をもっとも重視しているか（Feature Importance）を示します。

![Feature Importance](file:///""" + feat_imp_path + """)

- **分析結果**:
  - 最も重要視されている変数は何かをここで確認できます。
  - もし「霧」や「雨」といった悪天候要因が上位に来ていれば、**「天候によってセンサーを切り替える」という本来の目的を正しく学習できている証拠**となります。

## 2. 条件別のセンサー誤差（RMSE）比較

データセットを天候や距離といった条件ごとに分割し、各センサー単体とMeta-Learner（フュージョン後）のRMSE（二乗平均平方根誤差）を比較しました。単位はメートル(m)です。

| 条件 (Condition) | サンプル数 | AI単体 (RMSE) | カメラ単体 (RMSE) | LiDAR単体 (RMSE) | Meta-Learner (RMSE) |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

for row in results:
    md_content += f"| {row['Condition']} | {row['Samples']} | {row['AI']:.2f} | {row['Camera']:.2f} | {row['LiDAR']:.2f} | **{row['Meta-Learner']:.2f}** |\n"

md_content += """
> [!TIP]
> 上記の表から、**どの悪天候・距離の条件下でも、Meta-Learnerが単体センサーの誤差と同等かそれ以下に抑え込めているか**（リスクをヘッジできているか）を確認できます。

## 3. 平均的な予測重みの推移（条件別）

Meta-Learnerが各条件下で、平均してどのセンサーに高い重みを割り当てたかを集計しました。

| 条件 (Condition) | AIの平均重み | カメラの平均重み | LiDARの平均重み |
| :--- | :--- | :--- | :--- |
"""

for name, subset in conditions.items():
    if len(subset) == 0:
        continue
    w_ai_mean = subset['w_ai'].mean()
    w_cam_mean = subset['w_camera'].mean()
    w_lidar_mean = subset['w_lidar'].mean()
    md_content += f"| {name} | {w_ai_mean:.2f} | {w_cam_mean:.2f} | {w_lidar_mean:.2f} |\n"

md_content += """
- **透明性の確認**:
  - 例えば「濃霧」の条件下ではカメラの重みが下がり、LiDARやAIに重みが移っているか。
  - このような**「人間の直感（ドメイン知識）とモデルの出力傾向が一致しているか」**を見ることで、ブラックボックス性を排除し、安心してシステムに組み込むことができます。
"""

with open(md_path, 'w', encoding='utf-8') as f:
    f.write(md_content)

print("レポートの生成が完了しました。")
