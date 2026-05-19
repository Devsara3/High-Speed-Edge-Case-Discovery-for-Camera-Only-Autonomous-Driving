# -*- coding: utf-8 -*-
"""
CARLA Sensor Manager
センサーのスポーンとデータ変換（生データ -> Numpy配列/オブジェクト）を独立してカプセル化したモジュール。
Colabチュートリアルに基づき、RGBカメラに加えて、セマンティックセグメンテーション、LiDAR、Radar、IMU、GNSSに対応しました。
"""
import numpy as np
import weakref
import carla

class CarlaSensorManager:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.sensors = []
        
        # センサーごとの最新データを保持する辞書
        self.image_data = {'rgb_front': None, 'stereo_left': None, 'stereo_right': None}
        
        self.sensor_data = {
            'rgb_front': None,
            'seg_front': None,
            'lidar': None,
            'radar': None,
            'imu': None,
            'gnss': None
        }
        self.bp_library = self.world.get_blueprint_library()

    def spawn_rgb_camera(self, transform, role_name='rgb_front'):
        """単眼RGBカメラをスポーンして車両に取り付ける"""
        camera_bp = self.bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        self.sensors.append(camera)
        
        weak_self = weakref.ref(self)
        camera.listen(lambda image: CarlaSensorManager._parse_image(weak_self, image, role_name))
        return camera

    def spawn_semantic_segmentation_camera(self, transform, role_name='seg_front'):
        """セマンティックセグメンテーションカメラをスポーンして車両に取り付ける"""
        seg_bp = self.bp_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', '800')
        seg_bp.set_attribute('image_size_y', '600')
        seg_bp.set_attribute('fov', '90')
        
        seg_cam = self.world.spawn_actor(seg_bp, transform, attach_to=self.vehicle)
        self.sensors.append(seg_cam)
        
        weak_self = weakref.ref(self)
        seg_cam.listen(lambda image: CarlaSensorManager._parse_semantic_image(weak_self, image, role_name))
        return seg_cam

    def spawn_lidar(self, transform, role_name='lidar'):
        """LiDAR（レーザースキャナー）をスポーンして車両に取り付ける"""
        lidar_bp = self.bp_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50.0')                  # 50m範囲
        lidar_bp.set_attribute('channels', '32')                 # 32レイ
        lidar_bp.set_attribute('points_per_second', '56000')     # 点群生成レート
        lidar_bp.set_attribute('rotation_frequency', '10')       # 10Hz
        lidar_bp.set_attribute('sensor_tick', '0.05')            # 20 FPS (制御周期と同期)
        
        lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=self.vehicle)
        self.sensors.append(lidar)
        
        weak_self = weakref.ref(self)
        lidar.listen(lambda data: CarlaSensorManager._parse_lidar(weak_self, data, role_name))
        return lidar

    def spawn_radar(self, transform, role_name='radar'):
        """レーダーをスポーンして車両に取り付ける"""
        radar_bp = self.bp_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '30')
        radar_bp.set_attribute('range', '100.0')
        radar_bp.set_attribute('sensor_tick', '0.05')            # 20 FPS
        
        radar = self.world.spawn_actor(radar_bp, transform, attach_to=self.vehicle)
        self.sensors.append(radar)
        
        weak_self = weakref.ref(self)
        radar.listen(lambda data: CarlaSensorManager._parse_radar(weak_self, data, role_name))
        return radar

    def spawn_imu(self, transform, role_name='imu'):
        """IMU（慣性計測ユニット）をスポーンして車両に取り付ける"""
        imu_bp = self.bp_library.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.05')              # 20 FPS
        
        imu = self.world.spawn_actor(imu_bp, transform, attach_to=self.vehicle)
        self.sensors.append(imu)
        
        weak_self = weakref.ref(self)
        imu.listen(lambda data: CarlaSensorManager._parse_imu(weak_self, data, role_name))
        return imu

    def spawn_gnss(self, transform, role_name='gnss'):
        """GNSS（GPS受信機）をスポーンして車両に取り付ける"""
        gnss_bp = self.bp_library.find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', '0.05')              # 20 FPS
        
        gnss = self.world.spawn_actor(gnss_bp, transform, attach_to=self.vehicle)
        self.sensors.append(gnss)
        
        weak_self = weakref.ref(self)
        gnss.listen(lambda data: CarlaSensorManager._parse_gnss(weak_self, data, role_name))
        return gnss

    def spawn_stereo_cameras(self, base_transform, baseline=0.54):
        """ステレオカメラを同時に設置する"""
        left_transform = carla.Transform(
            carla.Location(x=base_transform.location.x, 
                           y=base_transform.location.y - baseline/2, 
                           z=base_transform.location.z),
            base_transform.rotation)
        self.spawn_rgb_camera(left_transform, role_name='stereo_left')

        right_transform = carla.Transform(
            carla.Location(x=base_transform.location.x, 
                           y=base_transform.location.y + baseline/2, 
                           z=base_transform.location.z),
            base_transform.rotation)
        self.spawn_rgb_camera(right_transform, role_name='stereo_right')

    @staticmethod
    def _parse_image(weak_self, image, role_name):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3] # BGR
        array = array[:, :, ::-1] # RGB
        
        self.image_data[role_name] = np.copy(array)
        self.sensor_data[role_name] = np.copy(array)

    @staticmethod
    def _parse_semantic_image(weak_self, image, role_name):
        self = weak_self()
        if not self:
            return
        # CityScapesパレットを適用して色つき画像に変換
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3] # BGR
        array = array[:, :, ::-1] # RGB
        
        self.sensor_data[role_name] = np.copy(array)

    @staticmethod
    def _parse_lidar(weak_self, data, role_name):
        self = weak_self()
        if not self:
            return
        # 1Dの浮動小数点バッファを [x, y, z, intensity] にリシェイプ
        lidar_points = np.frombuffer(data.raw_data, dtype=np.float32)
        lidar_points = np.reshape(lidar_points, (-1, 4))
        self.sensor_data[role_name] = lidar_points

    @staticmethod
    def _parse_radar(weak_self, data, role_name):
        self = weak_self()
        if not self:
            return
        # 1Dの浮動小数点バッファを [velocity, azimuth, altitude, depth] にリシェイプ
        radar_points = np.frombuffer(data.raw_data, dtype=np.float32)
        radar_points = np.reshape(radar_points, (-1, 4))
        self.sensor_data[role_name] = radar_points

    @staticmethod
    def _parse_imu(weak_self, data, role_name):
        self = weak_self()
        if not self:
            return
        self.sensor_data[role_name] = {
            'accel': (data.accelerometer.x, data.accelerometer.y, data.accelerometer.z),
            'gyro': (data.gyroscope.x, data.gyroscope.y, data.gyroscope.z),
            'compass': data.compass # ラジアン単位のヘディング角
        }

    @staticmethod
    def _parse_gnss(weak_self, data, role_name):
        self = weak_self()
        if not self:
            return
        self.sensor_data[role_name] = {
            'lat': data.latitude,
            'lon': data.longitude,
            'alt': data.altitude
        }

    def get_image(self, role_name='rgb_front'):
        """単眼カメラ等の画像を取得する"""
        if role_name in self.image_data:
            return self.image_data.get(role_name)
        return self.sensor_data.get(role_name)

    def get_sensor_data(self, role_name):
        """各種センサーの最新データを取得する"""
        return self.sensor_data.get(role_name)

    def destroy(self):
        """終了時に全センサーをクリーンアップする"""
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
