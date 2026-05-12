import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(tpe_csv_path, random_csv_path, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # データの読み込み
    try:
        df_tpe = pd.read_csv(tpe_csv_path)
        df_random = pd.read_csv(random_csv_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # 1. 時間対効果曲線 (Best score over trials)
    # TPEとRandomで、現在のトライアルまでに見つかった「最小スコア」を計算
    df_tpe['best_score'] = df_tpe['score'].cummin()
    df_random['best_score'] = df_random['score'].cummin()

    plt.figure(figsize=(10, 6))
    plt.plot(df_tpe['trial'], df_tpe['best_score'], label='Optuna (TPE) - Proposed', marker='o', markersize=4)
    plt.plot(df_random['trial'], df_random['best_score'], label='Random Search - Baseline', marker='x', markersize=4)
    
    plt.title('Time-Effectiveness Curve (Edge Case Search)')
    plt.xlabel('Number of Trials')
    plt.ylabel('Best (Minimum) Score (Lower is more critical)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_effectiveness_curve.png"))
    plt.close()

    # 2. 脆弱性マップ (Vulnerability Map / Parameter Scatter)
    # スコアが低い（危険な）領域を可視化。TPEの結果を使用
    plt.figure(figsize=(10, 8))
    # スコアに基づいて色付け。スコアが低いほど赤くする
    scatter = plt.scatter(
        df_tpe['sun_altitude_angle'], 
        df_tpe['fog_density'], 
        c=df_tpe['score'], 
        s=df_tpe['precipitation']*2 + 10, # 雨の強さを点の大きさで表現
        cmap='coolwarm',
        alpha=0.8
    )
    plt.colorbar(scatter, label='Detection Score (Lower = Higher Risk)')
    plt.title('Vulnerability Map (TPE Search)')
    plt.xlabel('Sun Altitude Angle (-15 to 90)')
    plt.ylabel('Fog Density (0 to 100)')
    
    # 注釈
    plt.text(0.05, 0.95, 'Size of points represents Precipitation', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vulnerability_map.png"))
    plt.close()
    
    print(f"Plots saved to '{output_dir}'")

if __name__ == "__main__":
    plot_results("results/history_tpe.csv", "results/history_random.csv")
