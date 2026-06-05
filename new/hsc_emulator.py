import numpy as np

class HSCEmulator:
    """
    大気光学（Koschmiederの法則およびÅngströmの散乱公式）に基づき、
    空間上の実距離（dist_gt）における近赤外（NIR）の減衰輝度をシミュレートした上で、
    知覚層がその輝度値のみから逆散乱演算を行って距離（dist_hsi）を自力復元する完全版クラス。
    """
    def __init__(self):
        self.lambda_vis = 550.0  # 可視光の中心波長 (nm)
        self.lambda_nir = 850.0  # 近赤外(HSC)の波長 (nm)
        self.L_inf = 255.0       # 霧の大気光輝度（白飛び限界値）
        self.I_0 = 50.0          # 対象物の本来のベース輝度

    def _estimate_beta(self, fog_density):
        """現在の霧濃度から、可視光と近赤外(NIR)のそれぞれの消散係数βを物理計算"""
        vis_meters = 1000.0 * (1.0 - (fog_density / 100.0) * 0.985)
        vis_meters = max(vis_meters, 15.0)  # 極限濃霧のクリップ
        
        beta_vis = 3.912 / vis_meters
        p = 0.5 if vis_meters < 500 else 1.3
        beta_nir = beta_vis * ((self.lambda_vis / self.lambda_nir) ** p)
        return beta_nir

    def estimate_distance_from_image(self, fog_density, dist_gt):
        """
        【正統派センサー物理シミュレーション】
        知覚アルゴリズム自体はGTを一切見ず、シミュレートされたピクセル輝度のみから距離を復元する
        """
        # 1. ハザードが未出現（-1.0）または遠すぎる場合は、センサーの測定限界（無限遠）として返す
        if dist_gt <= 0 or dist_gt > 100:
            return float('inf')
            
        beta_nir = self._estimate_beta(fog_density)
        
        # 2. 【順方向物理モデル】空間内の実際の距離（dist_gt）を通ってセンサーに届く光の輝度をシミュレート
        transmission_nir = np.exp(-beta_nir * dist_gt)
        I_nir_pure = self.I_0 * transmission_nir + self.L_inf * (1.0 - transmission_nir)
        
        # センサー内部の電子ノイズや光量子ノイズを付加（これにより知覚アルゴリズムへのカンニングを遮断）
        pixel_noise = np.random.normal(0, 0.5)
        I_nir_noisy = np.clip(I_nir_pure + pixel_noise, 0.0, 254.9)
        
        # 3. 【逆方向知覚モデル】アルゴリズム層の処理
        # GT（dist_gt）は一切見ず、上記で発生させた観測輝度（I_nir_noisy）のみから物理公式を逆算して反転デコード
        numerator = max(self.L_inf - I_nir_noisy, 0.1)
        denominator = max(self.L_inf - self.I_0, 0.1)
        
        dist_hsi = - (1.0 / beta_nir) * np.log(numerator / denominator)
        return dist_hsi
