import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def plot_3d_scatter(df, output_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(df['precipitation'], 
                    df['fog'], 
                    df['sun_altitude'], 
                    c=df['max_risk'], 
                    cmap='hot', 
                    s=50, 
                    alpha=0.8)
    
    ax.set_xlabel('Precipitation (%)')
    ax.set_ylabel('Fog Density (%)')
    ax.set_zlabel('Sun Altitude (deg)')
    ax.set_title('3D Critical Edge Case Domain')
    
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Max Risk Score')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Info] Saved 3D scatter plot to {output_path}")

def plot_2d_heatmap(df, output_path):
    # Bin the data to create a neat heatmap (e.g., 10% bins)
    df_binned = df.copy()
    df_binned['precip_bin'] = (df_binned['precipitation'] // 10 * 10).astype(int)
    df_binned['fog_bin'] = (df_binned['fog'] // 10 * 10).astype(int)
    
    # Calculate max risk per bin
    heatmap_data = df_binned.groupby(['fog_bin', 'precip_bin'])['max_risk'].max().reset_index()
    
    # Pivot for heatmap
    pivot_table = heatmap_data.pivot(index='fog_bin', columns='precip_bin', values='max_risk')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt=".2f", cbar_kws={'label': 'Max Risk Score'})
    
    plt.title('Vulnerability Heatmap (Precipitation vs Fog)')
    plt.xlabel('Precipitation (%)')
    plt.ylabel('Fog Density (%)')
    
    plt.gca().invert_yaxis() # Standardize origin to bottom-left if desired
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Info] Saved 2D heatmap to {output_path}")

def main():
    results_csv = 'phase4_results.csv'
    if not os.path.exists(results_csv):
        print(f"[Error] {results_csv} not found. Please run phase4_edge_case_discovery.py first.")
        return
        
    df = pd.read_csv(results_csv)
    
    if df.empty:
        print("[Error] No data in results CSV.")
        return
        
    os.makedirs('plots', exist_ok=True)
    
    plot_3d_scatter(df, 'plots/edge_case_3d_scatter.png')
    plot_2d_heatmap(df, 'plots/vulnerability_heatmap.png')
    
    print("\n[Success] All plots generated successfully in the 'plots' directory.")

if __name__ == '__main__':
    main()
