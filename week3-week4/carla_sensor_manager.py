# -*- coding: utf-8 -*-
"""
CARLA Sensor Manager
センサーのスポーンとデータ変換をカプセル化したモジュール。
貼り付けられたColabチュートリアルの正確なパラメータ設定（解像度1280x720、各更新レートなど）に完全に準拠して実装しています。
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
        """単眼RGBカメラをスポーンして車両に取り付ける (Tutorial 4.1に完全準拠)"""
        camera_bp = self.bp_library.find('sensor.camera.rgb')
        # チュートリアルと同一の設定
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.1')  # 10 Hz (0.1秒間隔)
        
        camera = self.world.spawn_actor(camera_bp, transform, attach_to=self.vehicle)
        self.sensors.append(camera)
        
        weak_self = weakref.ref(self)
        camera.listen(lambda image: CarlaSensorManager._parse_image(weak_self, image, role_name))
        return camera

    def spawn_semantic_segmentation_camera(self, transform, role_name='seg_front'):
        """セマンティックセグメンテーションカメラをスポーンして車両に取り付ける (Tutorial 4.2に完全準拠)"""
        seg_bp = self.bp_library.find('sensor.camera.semantic_segmentation')
        # チュートリアルと同一の設定 (RGBカメラのレートに合わせる)
        seg_bp.set_attribute('image_size_x', '1280')
        seg_bp.set_attribute('image_size_y', '720')
        seg_bp.set_attribute('fov', '90')
        seg_bp.set_attribute('sensor_tick', '0.1')
        
        seg_cam = self.world.spawn_actor(seg_bp, transform, attach_to=self.vehicle)
        self.sensors.append(seg_cam)
        
        weak_self = weakref.ref(self)
        seg_cam.listen(lambda image: CarlaSensorManager._parse_semantic_image(weak_self, image, role_name))
        return seg_cam

    def spawn_lidar(self, transform, role_name='lidar'):
        """LiDARをスポーンして車両に取り付ける (Tutorial 4.3に完全準拠)"""
        lidar_bp = self.bp_library.find('sensor.lidar.ray_cast')
        # チュートリアルと同一の設定
        lidar_bp.set_attribute('range', '50.0')                  # 50m範囲
        lidar_bp.set_attribute('channels', '32')                 # 32レイ
        lidar_bp.set_attribute('points_per_second', '56000')     # 点群生成レート
        lidar_bp.set_attribute('rotation_frequency', '10')       # 10Hz
        lidar_bp.set_attribute('sensor_tick', '0.1')             # 10Hz (0.1秒間隔)
        
        lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=self.vehicle)
        self.sensors.append(lidar)
        
        weak_self = weakref.ref(self)
        lidar.listen(lambda data: CarlaSensorManager._parse_lidar(weak_self, data, role_name))
        return lidar

    def spawn_radar(self, transform, role_name='radar'):
        """レーダーをスポーンして車両に取り付ける (Tutorial 4.4に完全準拠)"""
        radar_bp = self.bp_library.find('sensor.other.radar')
        # チュートリアルと同一の設定
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '30')
        radar_bp.set_attribute('range', '100')
        radar_bp.set_attribute('sensor_tick', '0.1')             # 10Hz (0.1秒間隔)
        
        radar = self.world.spawn_actor(radar_bp, transform, attach_to=self.vehicle)
        self.sensors.append(radar)
        
        weak_self = weakref.ref(self)
        radar.listen(lambda data: CarlaSensorManager._parse_radar(weak_self, data, role_name))
        return radar

    def spawn_imu(self, transform, role_name='imu'):
        """IMUをスポーンして車両に取り付ける (Tutorial 4.5に完全準拠)"""
        imu_bp = self.bp_library.find('sensor.other.imu')
        # チュートリアルと同一の設定
        imu_bp.set_attribute('sensor_tick', '0.05')              # 20Hz (0.05秒間隔)
        
        imu = self.world.spawn_actor(imu_bp, transform, attach_to=self.vehicle)
        self.sensors.append(imu)
        
        weak_self = weakref.ref(self)
        imu.listen(lambda data: CarlaSensorManager._parse_imu(weak_self, data, role_name))
        return imu

    def spawn_gnss(self, transform, role_name='gnss'):
        """GNSS (GPS) をスポーンして車両に取り付ける (Tutorial 4.6に完全準拠)"""
        gnss_bp = self.bp_library.find('sensor.other.gnss')
        # チュートリアルと同一の設定
        gnss_bp.set_attribute('sensor_tick', '0.5')               # 2Hz (0.5秒間隔)
        
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
        array = np.reshape(array, (image.height, image.width, 4)) # Keep as BGRA format
        
        self.image_data[role_name] = np.copy(array)
        self.sensor_data[role_name] = np.copy(array)

    @staticmethod
    def _parse_semantic_image(weak_self, image, role_name):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4)) # Keep as BGRA format
        
        self.sensor_data[role_name] = np.copy(array)

    @staticmethod
    def _parse_lidar(weak_self, data, role_name):
        self = weak_self()
        if not self:
            return
        lidar_points = np.frombuffer(data.raw_data, dtype=np.float32)
        lidar_points = np.reshape(lidar_points, (-1, 4))
        self.sensor_data[role_name] = lidar_points

    @staticmethod
    def _parse_radar(weak_self, data, role_name):
        self = weak_self()
        if not self:
            return
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
            'compass': data.compass
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
        if role_name in self.image_data:
            return self.image_data.get(role_name)
        return self.sensor_data.get(role_name)

    def get_sensor_data(self, role_name):
        return self.sensor_data.get(role_name)

    def destroy(self):
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
