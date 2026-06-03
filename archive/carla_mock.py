import cv2
import numpy as np

class MockCarlaEnv:
    """
    CARLAシミュレータの代替として機能するモック環境。
    時系列走行シミュレーションに対応し、Ego車両の前進や障害物との相対位置変化、
    歩行者の横断などを動的にシミュレートします。
    """
    def __init__(self, base_image_path):
        self.base_image = cv2.imread(base_image_path)
        if self.base_image is None:
            raise FileNotFoundError(f"Base image not found at {base_image_path}")
        
        # 初期化
        self.reset()

    def reset(self):
        """
        シミュレーション環境を初期化して、最初の状態に戻します。
        """
        self.time_step = 0
        self.ego_pos = [0.0, 0.0, 0.0]       # [x, y, z] (xが直進方向)
        self.ego_vel = [13.8, 0.0, 0.0]      # 初期速度 (約50km/h)
        self.traffic_light_color = 'red'     # デフォルト信号
        
        # 悪天候パラメータ (外部から設定される)
        self.sun_altitude_angle = 90.0
        self.precipitation = 0.0
        self.fog_density = 0.0
        
        # アクターの初期配置 (絶対座標を近く設定して、テスト時間を短縮)
        self.obstacles_def = {
            'pedestrian': {
                'class': 'pedestrian',
                'pos': [15.0, 3.0, 0.0],     # 15m前方、左側(Y=3.0)から横断予定
                'vel': [0.0, -1.5, 0.0],     # 横断速度 (Y軸負方向)
                'mu': 1.8,
                'active': False              # Egoが近づいたら動き出すフラグ
            },
            'construction_signal': {
                'class': 'construction_signal',
                'pos': [25.0, -2.0, 0.0],    # 25m前方、右側の路肩
                'vel': [0.0, 0.0, 0.0],
                'mu': 1.5
            },
            'car': {
                'class': 'car',
                'pos': [35.0, 0.0, 0.0],     # 35m前方
                'vel': [4.0, 0.0, 0.0],      # 先行車
                'mu': 1.0
            },
            'traffic_light': {
                'class': 'traffic_light',
                'pos': [45.0, 0.0, 5.0],     # 45m前方、上空5mに設置
                'vel': [0.0, 0.0, 0.0],
                'mu': 2.0
            }
        }
        
        # 現在のアクターの状態を複製
        self.obstacles = []
        for name, data in self.obstacles_def.items():
            self.obstacles.append({
                'class': data['class'],
                'pos': list(data['pos']),
                'vel': list(data['vel']),
                'mu': data['mu'],
                'active': data.get('active', True)
            })
            
        return self.get_image()

    def set_weather(self, sun_altitude_angle=90.0, precipitation=0.0, fog_density=0.0):
        """
        天候パラメータを設定します。
        """
        self.sun_altitude_angle = sun_altitude_angle
        self.precipitation = precipitation
        self.fog_density = fog_density

    def set_traffic_light_color(self, color):
        """
        信号の色を設定します。
        """
        if color in ['red', 'yellow', 'green']:
            self.traffic_light_color = color
            # 信号アクターの mu も動的に更新
            for obs in self.obstacles:
                if obs['class'] == 'traffic_light':
                    obs['color'] = color
                    obs['mu'] = 2.0 if color in ['red', 'yellow'] else 0.0
        else:
            raise ValueError(f"Invalid traffic light color: {color}")

    def step(self, action):
        """
        Ego車を前進させ、他アクターを動的に動かします。
        :param action: [throttle_or_brake, steer] のリストまたはNumPy配列
        """
        self.time_step += 1
        dt = 0.05 # 20 FPS (1ステップ0.05秒)
        
        if action is not None:
            accel = float(action[0])
            steer = float(action[1])
            
            # 簡易加減速物理モデル
            if accel < 0:
                self.ego_vel[0] += accel * 10.0 * dt  # 強い減速 (AEBなど)
            else:
                self.ego_vel[0] += accel * 3.0 * dt   # 緩やかな加速
            
            self.ego_vel[0] = max(0.0, self.ego_vel[0])
            
            # 操舵 (Y座標の変更)
            self.ego_pos[1] += steer * 2.0 * dt
        
        # Ego車の前進
        self.ego_pos[0] += self.ego_vel[0] * dt
        
        # アクターの動きを更新
        for obs in self.obstacles:
            if obs['class'] == 'pedestrian':
                # Egoが歩行者に接近 (残り10m以下) したら、歩行者が横断を開始する
                dist_to_ped = obs['pos'][0] - self.ego_pos[0]
                if dist_to_ped < 10.0:
                    obs['active'] = True
                    
                if obs.get('active', False):
                    # Y方向に横断
                    obs['pos'][1] += obs['vel'][1] * dt
                    # 道路の反対側まで渡りきったら静止
                    if obs['pos'][1] < -3.0:
                        obs['pos'][1] = -3.0
                        obs['vel'][1] = 0.0
                        
            elif obs['class'] == 'car':
                # 先行車も前進
                obs['pos'][0] += obs['vel'][0] * dt
                
        return self.get_image(), self.get_ground_truth()

    def get_image(self):
        """
        現在の走行位置・天候パラメータに基づき、前進感を演出した画像を返します。
        """
        img = self.base_image.copy().astype(np.float32)

        sun_alt = self.sun_altitude_angle
        fog_den = self.fog_density
        precip = self.precipitation

        # 1. 太陽高度（明るさ）エフェクト
        brightness_factor = max(0.1, min(1.0, (sun_alt + 15) / 105.0))
        img = img * brightness_factor

        # 2. 進行に伴う簡易的なズームイン・ズームアウト効果
        h, w, c = img.shape
        # 50m走る間に 1.0倍 〜 1.3倍までズームする簡易シミュレート
        zoom_factor = 1.0 + min(0.3, (self.ego_pos[0] / 50.0) * 0.3)
        if zoom_factor > 1.0:
            new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
            zoomed = cv2.resize(img, (new_w, new_h))
            dy, dx = (new_h - h) // 2, (new_w - w) // 2
            img = zoomed[dy:dy+h, dx:dx+w]

        # 3. 霧（Fog）エフェクト
        if fog_den > 0:
            fog_factor = fog_den / 100.0
            white_img = np.full_like(img, 255.0)
            img = cv2.addWeighted(img, 1.0 - (fog_factor * 0.8), white_img, fog_factor * 0.8, 0)

        # 4. 雨（Precipitation）エフェクト
        if precip > 0:
            rain_factor = precip / 100.0
            noise = np.random.normal(0, 20 * rain_factor, img.shape)
            img = img + noise
            blur_kernel = int(3 * rain_factor) * 2 + 1
            if blur_kernel > 1:
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                img = cv2.GaussianBlur(img_uint8, (blur_kernel, blur_kernel), 0).astype(np.float32)

        return np.clip(img, 0, 255).astype(np.uint8)

    def get_ground_truth(self):
        """
        現在の自車位置に対応したアクターたちの真値（Ground Truth）データを返します。
        """
        active_obstacles = []
        for obs in self.obstacles:
            # アクターまでの直進距離 (Egoの後方5mより前にあるものを対象とする)
            if obs['pos'][0] - self.ego_pos[0] >= -5.0:
                active_obstacles.append({
                    'class': obs['class'],
                    'pos': list(obs['pos']),
                    'vel': list(obs['vel']),
                    'mu': obs['mu'],
                    'color': obs.get('color', self.traffic_light_color) if obs['class'] == 'traffic_light' else None
                })
                
        return {
            'ego_pos': list(self.ego_pos),
            'ego_vel': list(self.ego_vel),
            'obstacles': active_obstacles
        }
