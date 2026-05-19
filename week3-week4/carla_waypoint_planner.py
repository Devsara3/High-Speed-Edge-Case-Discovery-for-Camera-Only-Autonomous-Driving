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
        
    def get_target_waypoint(self, lookahead_distance=5.0):
        """
        車両の現在位置から、指定した距離（lookahead_distance）だけ先の
        車線中央の目標ポイント（Waypoint）を取得する。
        ※ 将来的にGlobalRoutePlannerなどを導入すれば、ここを「次のナビゲーションポイント」に差し替えるだけで済む。
        """
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        
        # 現在位置に最も近いマップ上のWaypoint（車線中心）を取得
        current_wp = self.map.get_waypoint(vehicle_location)
        
        # 車線の流れに沿って前方に進んだ先のWaypointを取得
        next_wps = current_wp.next(lookahead_distance)
        if len(next_wps) == 0:
            return current_wp # 行き止まりの場合は現在地を返す
            
        return next_wps[0]
        
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
