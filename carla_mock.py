import cv2
import numpy as np

class MockCarlaEnv:
    """
    CARLAシミュレータの代替として機能するモック環境。
    ベース画像に対して天候パラメータ（太陽角度、雨、霧）に応じた画像処理を行い、
    悪天候時の車載カメラ画像を擬似的に生成します。
    """
    def __init__(self, base_image_path):
        self.base_image = cv2.imread(base_image_path)
        if self.base_image is None:
            raise FileNotFoundError(f"Base image not found at {base_image_path}")
        
        # デフォルト天候
        self.sun_altitude_angle = 90.0
        self.precipitation = 0.0
        self.fog_density = 0.0

    def set_weather(self, sun_altitude_angle=90.0, precipitation=0.0, fog_density=0.0):
        """
        天候パラメータを設定します。
        :param sun_altitude_angle: 太陽の高度 (-90.0 〜 90.0)。低いと暗くなる/逆光
        :param precipitation: 雨の強さ (0.0 〜 100.0)
        :param fog_density: 霧の濃さ (0.0 〜 100.0)
        """
        self.sun_altitude_angle = sun_altitude_angle
        self.precipitation = precipitation
        self.fog_density = fog_density

    def get_image(self):
        """
        現在の天候パラメータに基づいてベース画像にエフェクトをかけた画像を返します。
        """
        img = self.base_image.copy().astype(np.float32)

        # 1. 太陽高度（明るさ）エフェクト
        # sun_altitude_angle: 90が最も明るく、0以下は夜（真っ暗）
        # 簡単のため、-15度以下はほぼ見えないようにする
        brightness_factor = max(0.1, min(1.0, (self.sun_altitude_angle + 15) / 105.0))
        img = img * brightness_factor

        # 2. 霧（Fog）エフェクト
        # fog_density: 画像全体を白っぽくし、コントラストを下げる
        if self.fog_density > 0:
            fog_factor = self.fog_density / 100.0
            white_img = np.full_like(img, 255.0)
            img = cv2.addWeighted(img, 1.0 - (fog_factor * 0.8), white_img, fog_factor * 0.8, 0)

        # 3. 雨（Precipitation）エフェクト
        # 画像にランダムなノイズ（雨粒）と全体的なぼかしを追加
        if self.precipitation > 0:
            rain_factor = self.precipitation / 100.0
            
            # ガウシアンノイズの追加
            noise = np.random.normal(0, 20 * rain_factor, img.shape)
            img = img + noise

            # ぼかしの追加 (水滴による視界不良のシミュレート)
            blur_kernel = int(3 * rain_factor) * 2 + 1 # 1, 3, 5, 7...
            if blur_kernel > 1:
                # float32なので一時的にuint8に変換してぼかす
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                img = cv2.GaussianBlur(img_uint8, (blur_kernel, blur_kernel), 0).astype(np.float32)

        return np.clip(img, 0, 255).astype(np.uint8)

    def get_ground_truth(self):
        """
        モック画像のシーンに対応する、固定の真値（Ground Truth）データを返します。
        本番のCARLA環境では、ここでAPIから動的にegoやtargetの情報を取得します。
        """
        return {
            'ego_pos': [0.0, 0.0, 0.0],       # 自車位置
            'ego_vel': [13.8, 0.0, 0.0],      # 自車速度 (約50km/hで前進)
            'target_pos': [15.0, 0.0, 0.0],   # 相手車両位置 (15m前方)
            'target_vel': [0.0, 0.0, 0.0],    # 相手車両速度 (停止中)
            'target_class': 'car'             # 相手車両のクラス
        }

if __name__ == "__main__":
    # テスト実行
    env = MockCarlaEnv("base_image.png")
    
    # テスト画像保存用ディレクトリ
    import os
    os.makedirs("test_outputs", exist_ok=True)
    
    cv2.imwrite("test_outputs/clear.jpg", env.get_image())
    
    env.set_weather(sun_altitude_angle=-5.0)
    cv2.imwrite("test_outputs/dark.jpg", env.get_image())
    
    env.set_weather(fog_density=80.0)
    cv2.imwrite("test_outputs/foggy.jpg", env.get_image())
    
    env.set_weather(precipitation=100.0)
    cv2.imwrite("test_outputs/rainy.jpg", env.get_image())
    
    print("Test images generated in 'test_outputs/'")
