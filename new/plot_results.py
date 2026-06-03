import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scenic # ※ローカル固有のライブラリ（またはtypo）の可能性が高いため、ModuleNotFoundErrorを避けるためコメントアウトしています。必要に応じて戻してください。

def plot_3d_scatter(df):
    """
    図1: 気象条件(雨・霧・日射角)とリスクスコアの3D散布図
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1試行ごとの最大リスクをプロットするため、まずはグループ化
    df_trial = df.groupby(['precipitation', 'fog', 'sun_altitude'])['risk_score'].max().reset_index()
    
    sc = ax.scatter(
        df_trial['precipitation'], 
        df_trial['fog'], 
        df_trial['sun_altitude'], 
        c=df_trial['risk_score'], 
        cmap='plasma', 
        s=df_trial['risk_score'] * 2.0, # リスクが高いほど点を大きく
        alpha=0.8, 
        edgecolors='w', 
        linewidths=0.5
    )
    
    ax.set_xlabel('Precipitation (%)', fontsize=11, labelpad=10)
    ax.set_ylabel('Fog Density (%)', fontsize=11, labelpad=10)
    ax.set_zlabel('Sun Altitude (deg)', fontsize=11, labelpad=10)
    ax.set_title('3D Weather Space Exploration for Critical Edge Cases', fontsize=13, pad=15)
    
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Maximum Risk Score $R(t)$', fontsize=11)
    
    plt.savefig('new/figure1_3d_weather_risk_scatter.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] 図1 (3D散布図): 'new/figure1_3d_weather_risk_scatter.png' を保存しました。")
    plt.close()

def plot_2d_heatmap(df):
    """
    図2: 降水量×霧の2Dヒートマップ (天候デジタイズ平面)
    """
    # 10%刻みのグリッドごとの最大リスクスコアの平均値をピボット化
    df['precip_grid'] = np.round(df['precipitation'] / 10.0) * 10.0
    df['fog_grid'] = np.round(df['fog'] / 10.0) * 10.0
    
    pivot_table = df.pivot_table(
        values='risk_score', 
        index='fog_grid', 
        columns='precip_grid', 
        aggfunc='max' # 最悪値をあぶり出すためmaxを採用
    ).sort_index(ascending=False) # 縦軸を上に向かって濃くする
    
    plt.figure(figsize=(9, 7))
    
    # 論文用のクリーンなカラーマップで描写
    im = plt.imshow(pivot_table.values, cmap='YlOrRd', extent=[
        pivot_table.columns.min()-5, pivot_table.columns.max()+5,
        pivot_table.index.min()-5, pivot_table.index.max()+5
    ], aspect='auto')
    
    # グリッドの数値ラベルをマスの中にプロット
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            if not np.isnan(val):
                plt.text(pivot_table.columns[j], pivot_table.index[i], f'{val:.1f}', 
                         ha='center', va='center', color='black' if val < pivot_table.values.max()*0.6 else 'white', fontsize=9)

    plt.xlabel('Precipitation (%)', fontsize=12)
    plt.ylabel('Fog Density (%)', fontsize=12)
    plt.title('2D Critical Risk Heatmap (Precipitation vs Fog)', fontsize=13, pad=15)
    
    # 目盛りの調整
    plt.xticks(np.arange(0, 101, 20))
    plt.yticks(np.arange(0, 101, 20))
    
    cbar = plt.colorbar(im)
    cbar.set_label('Peak Risk Score $R(t)$', fontsize=12)
    plt.grid(False)
    
    plt.savefig('new/figure2_2d_weather_risk_heatmap.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] 図2 (2Dヒートマップ): 'new/figure2_2d_weather_risk_heatmap.png' を保存しました。")
    plt.close()

def main():
    csv_path = 'new/fused_risk_timeseries.csv'
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} が存在しません。先に phase4_edge_case_discovery.py を実行してください。")
        return
        
    df = pd.read_csv(csv_path)
    
    # 描画スタイルの設定（論文用のスマートなフォント配置）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plot_3d_scatter(df)
    plot_2d_heatmap(df)

if __name__ == '__main__':
    main()
