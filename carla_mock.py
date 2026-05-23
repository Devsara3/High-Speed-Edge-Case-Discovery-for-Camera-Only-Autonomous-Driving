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
        self.traffic_light_color = 'red'  # 'red', 'yellow', 'green'

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

    def set_traffic_light_color(self, color):
        """
        信号の色を設定します。
        :param color: 'red', 'yellow', 'green'
        """
        if color in ['red', 'yellow', 'green']:
            self.traffic_light_color = color
        else:
            raise ValueError(f"Invalid traffic light color: {color}")

    def get_image(self):
        """
        現在の天候パラメータに基づいてベース画像にエフェクトをかけた画像を返します。
        """
        img = self.base_image.copy().astype(np.float32)

        sun_alt = self.sun_altitude_angle
        fog_den = self.fog_density
        precip = self.precipitation

        # 1. 太陽高度（明るさ）エフェクト
        brightness_factor = max(0.1, min(1.0, (sun_alt + 15) / 105.0))
        img = img * brightness_factor

        # 2. 霧（Fog）エフェクト
        if fog_den > 0:
            fog_factor = fog_den / 100.0
            white_img = np.full_like(img, 255.0)
            img = cv2.addWeighted(img, 1.0 - (fog_factor * 0.8), white_img, fog_factor * 0.8, 0)

        # 3. 雨（Precipitation）エフェクト
        if precip > 0:
            rain_factor = precip / 100.0
            
            # ガウシアンノイズの追加
            noise = np.random.normal(0, 20 * rain_factor, img.shape)
            img = img + noise

            # ぼかしの追加 (水滴による視界不良のシミュレート)
            blur_kernel = int(3 * rain_factor) * 2 + 1 # 1, 3, 5, 7...
            if blur_kernel > 1:
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                img = cv2.GaussianBlur(img_uint8, (blur_kernel, blur_kernel), 0).astype(np.float32)

        return np.clip(img, 0, 255).astype(np.uint8)

    def get_ground_truth(self):
        """
        モック画像のシーンに対応する、複数のオブジェクトの真値（Ground Truth）データを返します。
        """
        tl_mu = 2.0 if self.traffic_light_color in ['red', 'yellow'] else 0.0
        return {
            'ego_pos': [0.0, 0.0, 0.0],       # 自車位置
            'ego_vel': [13.8, 0.0, 0.0],      # 自車速度 (約50km/hで前進)
            'obstacles': [
                {
                    'class': 'pedestrian',
                    'pos': [10.0, 1.5, 0.0],   # 10m前方
                    'vel': [0.0, -1.0, 0.0],   # 横断速度
                    'mu': 1.8
                },
                {
                    'class': 'car',
                    'pos': [25.0, 0.0, 0.0],   # 25m前方
                    'vel': [10.0, 0.0, 0.0],   # 先行車速度
                    'mu': 1.0
                },
                {
                    'class': 'construction_signal', # 静止標識/バリケード
                    'pos': [18.0, -2.0, 0.0],
                    'vel': [0.0, 0.0, 0.0],
                    'mu': 1.5
                },
                {
                    'class': 'traffic_light',
                    'pos': [20.0, 0.0, 5.0],
                    'vel': [0.0, 0.0, 0.0],
                    'color': self.traffic_light_color,
                    'mu': tl_mu
                }
            ]
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
