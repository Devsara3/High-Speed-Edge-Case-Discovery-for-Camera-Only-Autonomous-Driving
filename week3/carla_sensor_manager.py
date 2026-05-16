"""
CARLA Sensor Manager
将来的な拡張性（複数カメラ、ステレオカメラ、LiDARなどの追加）を考慮し、
センサーのスポーンとデータ変換（生データ -> AI用Numpy配列）を独立してカプセル化したモジュール。
"""
import numpy as np
import weakref

class CarlaSensorManager:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.sensors = []
        # カメラごとの最新画像を保持する辞書
        self.image_data = {'rgb_front': None, 'stereo_left': None, 'stereo_right': None}
        self.bp_library = self.world.get_blueprint_library()

    def spawn_rgb_camera(self, transform, role_name='rgb_front'):
        """単眼RGBカメラをスポーンして車両に取り付ける"""
        camera_bp = self.bp_library.find('sensor.camera.rgb')
        # 画像サイズと視野角（FOV）の設定
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        self.sensors.append(camera)
        
        # コールバックの登録 (weakrefを使用してメモリリークを防ぐ)
        weak_self = weakref.ref(self)
        camera.listen(lambda image: CarlaSensorManager._parse_image(weak_self, image, role_name))
        return camera

    def spawn_stereo_cameras(self, base_transform, baseline=0.54):
        """
        ステレオカメラ（左右2つのカメラ）を同時に設置する。
        base_transform: 車両の中心となる設置位置
        baseline: 左右のカメラ間の距離（メートル）
        """
        import carla
        # 左カメラ (中心から左に baseline/2 移動)
        left_transform = carla.Transform(
            carla.Location(x=base_transform.location.x, 
                           y=base_transform.location.y - baseline/2, 
                           z=base_transform.location.z),
            base_transform.rotation)
        self.spawn_rgb_camera(left_transform, role_name='stereo_left')

        # 右カメラ (中心から右に baseline/2 移動)
        right_transform = carla.Transform(
            carla.Location(x=base_transform.location.x, 
                           y=base_transform.location.y + baseline/2, 
                           z=base_transform.location.z),
            base_transform.rotation)
        self.spawn_rgb_camera(right_transform, role_name='stereo_right')

    @staticmethod
    def _parse_image(weak_self, image, role_name):
        """CARLAから送られてくる生データ(BGRA)を、AIで扱いやすいRGBのNumpy配列に変換する"""
        self = weak_self()
        if not self:
            return
            
        # CARLAの画像をNumpy配列に変換
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4)) # BGRA形式で取得される
        array = array[:, :, :3] # Alphaチャンネルを削ってBGRにする
        array = array[:, :, ::-1] # OpenCVやPyTorchで扱いやすいようにBGRからRGBへ反転
        
        self.image_data[role_name] = np.copy(array)

    def get_image(self, role_name='rgb_front'):
        """指定したカメラの最新画像を取得する"""
        return self.image_data.get(role_name)

    def destroy(self):
        """終了時に全センサーをクリーンアップする"""
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
