"""
CARLA Waypoint Planner
将来的に複雑な経路探索（A*アルゴリズムや交差点でのナビゲーション機能）を追加できるよう、
ルート計算や偏差（エラー）の算出をPID制御から分離したモジュールです。
"""
import math

class CarlaWaypointPlanner:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.map = self.world.get_map()
        
    def get_target_waypoint(self, lookahead_distance=5.0, obstacle=None):
        """
        車両の現在位置から、指定した距離（lookahead_distance）だけ先の
        車線中央の目標ポイント（Waypoint）を取得する。
        障害物（obstacle）が指定されている場合、自動的に左側へ回避するルート（車線変更）を計算する。
        """
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        
        # 現在位置に最も近いマップ上のWaypoint（車線中心）を取得
        current_wp = self.map.get_waypoint(vehicle_location)
        
        # 車線の流れに沿って前方に進んだ先のWaypointを取得
        next_wps = current_wp.next(lookahead_distance)
        if len(next_wps) == 0:
            target_wp = current_wp
        else:
            target_wp = next_wps[0]
            
        # 静的障害物がある場合の「避ける（車線変更）」制御ロジック
        if obstacle and obstacle.is_alive:
            obs_transform = obstacle.get_transform()
            obs_loc = obs_transform.location
            
            # 自車から障害物へのベクトル
            vec = obs_loc - vehicle_location
            fwd = vehicle_transform.get_forward_vector()
            
            # 進行方向の距離 (long_dist) を算出
            long_dist = vec.x * fwd.x + vec.y * fwd.y + vec.z * fwd.z
            
            # 回避（よける）制御の適用判定：前方25mから通過後15mまで
            offset_val = 0.0
            if 0.0 <= long_dist <= 25.0:
                # 障害物が前方25m以内にあるとき、右側へ3.5m（1車線分）回避する
                # 挙動の急激な変化を防ぐため、前方25m〜15mの間で徐々にオフセットを増やす
                if long_dist > 15.0:
                    offset_val = 3.5 * (1.0 - (long_dist - 15.0) / 10.0)
                else:
                    offset_val = 3.5
            elif -15.0 <= long_dist < 0.0:
                # 障害物を通過した後、15mかけて元の車線に滑らかに戻る
                offset_val = 3.5 * (1.0 - (abs(long_dist) / 15.0))
                
            if abs(offset_val) > 0.01:
                # 目標位置を右側（右ベクトルのプラス方向）にオフセット
                right_vector = target_wp.transform.get_right_vector()
                target_loc = target_wp.transform.location + right_vector * offset_val
                
                # CARLAのWaypointのLocationは書き換え不可のため、ダミーのTargetPointクラスを作成して返す
                class TargetPoint:
                    class Transform:
                        class Location:
                            def __init__(self, x, y, z):
                                self.x = x
                                self.y = y
                                self.z = z
                        def __init__(self, loc, rot):
                            self.location = self.Location(loc.x, loc.y, loc.z)
                            self.rotation = rot
                    def __init__(self, loc, rot):
                        self.transform = self.Transform(loc, rot)
                        
                return TargetPoint(target_loc, target_wp.transform.rotation)
                
        return target_wp
        
    def calculate_steering_error(self, target_waypoint):
        """
        目標のWaypointに向かうための、車両の現在の向きとの角度のズレ（エラー）を計算する。
        このエラー値をPID（横方向）に渡すことで、車線をキープして走るようになる。
        """
        vehicle_transform = self.vehicle.get_transform()
        
        # 車両の現在の向き（Yaw）
        vehicle_yaw = vehicle_transform.rotation.yaw
        
        # 車両から目標ポイントへのベクトル（X, Yの差分）
        dx = target_waypoint.transform.location.x - vehicle_transform.location.x
        dy = target_waypoint.transform.location.y - vehicle_transform.location.y
        
        # 目標への絶対角度を計算
        target_yaw = math.degrees(math.atan2(dy, dx))
        
        # 角度のズレ（エラー）を計算し、-180度 〜 180度の範囲に正規化する
        error_yaw = target_yaw - vehicle_yaw
        while error_yaw > 180.0:
            error_yaw -= 360.0
        while error_yaw < -180.0:
            error_yaw += 360.0
            
        # PIDコントローラーが扱いやすいようにラジアンに変換して返す
        return math.radians(error_yaw)

    def get_target_speed(self):
        """
        将来的に、カーブの手前で減速したり、赤信号や障害物をAIが認識して速度を0にする
        ロジックをここに入れるための箱です。
        現在は固定値（市街地想定の10m/s = 36km/h）を返します。
        """
        return 10.0
