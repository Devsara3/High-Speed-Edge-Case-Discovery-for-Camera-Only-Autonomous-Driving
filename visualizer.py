import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def _find_csv(output_dir, suffix):
    """自动检测 Mock (history_tpe_*.csv) 或 CARLA 真实 (real_history_TPE_*.csv) 结果文件"""
    candidates = [
        os.path.join(output_dir, f"real_history_TPE_{suffix}.csv"),
        os.path.join(output_dir, f"real_history_Random_{suffix}.csv"),
        os.path.join(output_dir, f"history_tpe_{suffix}.csv"),
        os.path.join(output_dir, f"history_random_{suffix}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # 兜底：通配符匹配
    for pattern in [f"real_history_*_{suffix}.csv", f"history_*_{suffix}.csv"]:
        matches = glob.glob(os.path.join(output_dir, pattern))
        if matches:
            return matches[0]
    return None

def plot_results(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    tpe_red_path = _find_csv(output_dir, "red")
    tpe_green_path = _find_csv(output_dir, "green")
    
    if tpe_red_path is None:
        print(f"Error: 未找到 results/history_*_red.csv 或 results/real_history_*_red.csv")
        print("请先运行 carla_optuna_optimizer.py 或 optimizer.py")
        return

    print(f"使用数据: {tpe_red_path}")
    if tpe_green_path:
        print(f"使用数据: {tpe_green_path}")
        
    df_red = pd.read_csv(tpe_red_path)
    
    # グラフのスタイル設定 (Seabornのホワイトグリッド)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    sns.set_context("talk")
    
    # ------------------ 1. 脆弱性（知覚ギャップ）の推移（赤信号） ------------------
    plt.figure(figsize=(12, 6))
    plt.plot(df_red['trial'], df_red['gap'], label='Perception Gap (Vulnerability)', color='#e056fd', marker='o', linewidth=2)
    
    plt.title('Vulnerability (Perception Gap) Trend: Standard RGB Camera (Red Light Scenario)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Optimization Trials', fontsize=12)
    plt.ylabel('Perception Gap (Vulnerability)', fontsize=12)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vulnerability_trend.png"), dpi=150)
    plt.close()
    
    # ------------------ 2. パラメータ空間における脆弱性マップ ------------------
    plt.figure(figsize=(12, 8))
    
    sc = plt.scatter(
        df_red['sun_altitude_angle'], 
        df_red['fog_density'], 
        c=df_red['gap'], 
        s=df_red['precipitation']*3 + 10,
        cmap='plasma',
        alpha=0.8,
        edgecolors='w',
        linewidth=0.5
    )
    plt.colorbar(sc, label='Perception Gap (Vulnerability)')
    plt.title('Parameter Space Vulnerability Map (Point Size = Precipitation)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Sun Altitude Angle (deg)', fontsize=12)
    plt.ylabel('Fog Density (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "vulnerability_map.png"), dpi=150)
    plt.close()
    
    # ------------------ 3. 信号機の色（赤 vs 青/緑）による影響の箱ひげ図 ------------------
    if tpe_green_path and os.path.exists(tpe_green_path):
        df_green = pd.read_csv(tpe_green_path)
        
        plt.figure(figsize=(10, 6))
        # 箱ひげ図で比較
        sns.boxplot(data=[df_red['gap'], df_green['gap']], palette=['#ff7675', '#55efc4'])
        plt.xticks([0, 1], ['Red Traffic Light', 'Green Traffic Light'])
        plt.ylabel('Perception Gap (Vulnerability)', fontsize=12)
        plt.title('Impact of Traffic Light Color on RGB Safety Risk', fontsize=16, fontweight='bold', pad=15)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "traffic_light_color_impact.png"), dpi=150)
        plt.close()
        
    print(f"[SUCCESS] Visualizations saved to '{output_dir}/'")

if __name__ == "__main__":
    plot_results()
