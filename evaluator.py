import cv2
import numpy as np
from ultralytics import YOLO

class YoloEvaluator:
    """
    YOLOv8を用いて画像中の物体を検出し、認識精度（スコア）を評価するクラス。
    """
    def __init__(self, model_name='yolov8n.pt'):
        """
        モデルの初期化。初回はインターネットからダウンロードされます。
        """
        self.model = YOLO(model_name)
        # 車や歩行者など、注目するクラスID (COCO dataset: 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck, 9:traffic light, 11:stop sign)
        self.target_classes = [0, 1, 2, 3, 5, 7, 9, 11]

        # 深層学習深度推定モデル (MiDaS) のロード試行 (実用的な単眼3D検出システムとして機能させます)
        try:
            import sys
            import os
            # 親ディレクトリの同階層にある carla_edge_case_search を sys.path に追加して week3 を読み込めるようにする
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            search_path = os.path.join(parent_dir, "carla_edge_case_search")
            if os.path.exists(search_path) and search_path not in sys.path:
                sys.path.append(search_path)
            
            from week3.depth_estimation import MiDaS_DepthEstimator
            # 軽量なMiDaS_smallをロード
            self.depth_estimator = MiDaS_DepthEstimator(model_type="MiDaS_small")
            print("[INFO] Real-world Deep Learning Depth Estimator (MiDaS) loaded successfully!")
        except Exception as e:
            self.depth_estimator = None
            print(f"[WARNING] Could not load MiDaS Depth Estimator. Falling back to Geometric Pinhole Model. Error: {e}")

    def detect_traffic_light_color(self, crop):
        """
        切り抜いた信号画像からHSV色相情報を基に信号機の色(red, yellow, green, unknown)を判定します。
        """
        if crop is None or crop.size == 0:
            return 'unknown'
        
        # HSVに変換
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # 各色のHSV範囲定義 (OpenCVのHue範囲は0-180)
        # 赤は0-10付近と170-180付近の2カ所にある
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 70])
        upper_red2 = np.array([180, 255, 255])
        
        # 黄色
        lower_yellow = np.array([15, 70, 70])
        upper_yellow = np.array([35, 255, 255])
        
        # 緑（青信号）
        lower_green = np.array([36, 70, 70])
        upper_green = np.array([90, 255, 255])
        
        # マスク作成
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # ピクセル数のカウント
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels = cv2.countNonZero(mask_green)
        
        total_pixels = crop.shape[0] * crop.shape[1]
        threshold_ratio = 0.02 # 2%以上のピクセルが該当色なら有効とする
        
        counts = {
            'red': red_pixels,
            'yellow': yellow_pixels,
            'green': green_pixels
        }
        
        best_color = max(counts, key=counts.get)
        if counts[best_color] > total_pixels * threshold_ratio:
            return best_color
        
        return 'unknown'

    def evaluate_multi(self, image, return_image=False):
        """
        画像を推論し、検出したすべてのオブジェクトの詳細リストを返します。
        """
        processed_image = image.copy()
        results = self.model(processed_image, verbose=False, conf=0.1)
        detections = []
        
        # クラスIDのマッピング
        class_mapping = {
            0: 'pedestrian',
            1: 'car', 2: 'car', 3: 'car', 5: 'car', 7: 'car',
            9: 'traffic_light',
            11: 'construction_signal'
        }
        
        annotated_frame = processed_image.copy() if return_image else None
        
        # カメラパラメータの動的計算 (CARLAの視野角FOV=110度を想定)
        img_width = float(processed_image.shape[1])
        img_height = float(processed_image.shape[0])
        fov_rad = np.radians(110.0)
        focal_length = img_width / (2.0 * np.tan(fov_rad / 2.0))
        c_x = img_width / 2.0
        c_y = img_height / 2.0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = class_mapping.get(cls_id, 'unknown')
                    
                    # 信号の色判別
                    tl_color = None
                    if class_name == 'traffic_light':
                        crop = processed_image[y1:y2, x1:x2]
                        tl_color = self.detect_traffic_light_color(crop)
                    
                    # Z距離の推定 (MiDaS or Pinhole)
                    if hasattr(self, 'depth_estimator') and self.depth_estimator is not None:
                        rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                        depth_map = self.depth_estimator.estimate(rgb_image)
                        
                        h, w = depth_map.shape
                        bx1, by1 = max(0, x1), max(0, y1)
                        bx2, by2 = min(w - 1, x2), min(h - 1, y2)
                        
                        if bx2 > bx1 and by2 > by1:
                            box_depth = depth_map[by1:by2, bx1:bx2]
                            median_disparity = np.median(box_depth)
                            if median_disparity > 0:
                                z_dist = 200.0 / median_disparity
                            else:
                                z_dist = float('inf')
                        else:
                            z_dist = float('inf')
                    else:
                        w_pixel = float(box.xywh[0][2])
                        
                        # 1. ピンホール幅モデルによる距離推定 (z_dist_width)
                        real_width = 1.8
                        if class_name == 'pedestrian':
                            real_width = 0.5
                        elif class_name == 'traffic_light':
                            real_width = 0.3
                        elif class_name == 'construction_signal':
                            real_width = 0.8
                            
                        if w_pixel > 1.0:
                            z_dist_width = (focal_length * real_width) / w_pixel
                        else:
                            z_dist_width = float('inf')

                        # 2. 接地面制約モデル (Ground Plane Constraint) による距離推定 (z_dist_ground)
                        # カメラの高さ H = 1.4m、ピッチ角 pitch = -5.0度 (下向き) を想定
                        H_cam = 1.4
                        pitch_rad = np.radians(-5.0)
                        
                        y2_val = float(y2)
                        
                        # 俯角 phi の計算 (画像中心からの偏角 + カメラピッチ角)
                        angle_from_center = np.arctan((y2_val - c_y) / focal_length)
                        phi = pitch_rad + angle_from_center
                        
                        # 自車からの水平距離の算出
                        # カメラが下を向いているため、phi < 0 のときに路面と交差する
                        if phi < -1e-3:
                            z_dist_ground = H_cam / np.tan(-phi)
                        else:
                            z_dist_ground = float('inf')

                        # 3. ハイブリッド距離の決定
                        if class_name == 'traffic_light':
                            # 信号機は空中に浮いているため、接地面モデルは適用せず幅モデルを100%採用
                            z_dist = z_dist_width
                        else:
                            # どちらかが異常値の場合は他方を採用
                            if np.isinf(z_dist_width) or z_dist_width > 150.0:
                                z_dist = z_dist_ground
                            elif np.isinf(z_dist_ground) or z_dist_ground > 150.0 or z_dist_ground < 1.0:
                                z_dist = z_dist_width
                            else:
                                # 両方とも妥当な値の場合は、重み付け平均
                                z_dist = 0.4 * z_dist_width + 0.6 * z_dist_ground
                            
                    # 逆投影によるカメラ座標基準の相対3D座標 [X, Y, Z] の算出
                    if not np.isinf(z_dist):
                        u_c = float(box.xywh[0][0])
                        v_c = float(box.xywh[0][1])
                        x_pred = (u_c - c_x) * z_dist / focal_length
                        y_pred = (v_c - c_y) * z_dist / focal_length
                        yolo3d_rel_pos = [x_pred, y_pred, z_dist]
                    else:
                        yolo3d_rel_pos = None

                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'z_distance': z_dist,
                        'yolo3d_rel_pos': yolo3d_rel_pos,
                        'traffic_light_color': tl_color,
                        'bbox': (x1, y1, x2, y2)
                    })
                    
                    if return_image:
                        color = (0, 255, 0)
                        if class_name == 'traffic_light' and tl_color is not None:
                            if tl_color == 'red':
                                color = (0, 0, 255)
                            elif tl_color == 'yellow':
                                color = (0, 255, 255)
                            elif tl_color == 'green':
                                color = (0, 255, 0)
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name} {z_dist:.1f}m {conf:.2f}"
                        if tl_color:
                            label += f" [{tl_color}]"
                        cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    
        if return_image:
            return detections, annotated_frame
        return detections

    def evaluate(self, image, return_image=False):
        """
        単一オブジェクト用インターフェースへの後方互換性を持つラッパーメソッド。
        """
        res = self.evaluate_multi(image, return_image=return_image)
        if return_image:
            detections, annotated = res
            min_z = float('inf')
            total_conf = 0.0
            for d in detections:
                if d['z_distance'] < min_z:
                    min_z = d['z_distance']
                total_conf += d['confidence']
            return min_z, total_conf, annotated
        else:
            detections = res
            min_z = float('inf')
            total_conf = 0.0
            for d in detections:
                if d['z_distance'] < min_z:
                    min_z = d['z_distance']
                total_conf += d['confidence']
            return min_z, total_conf

if __name__ == "__main__":
    from carla_mock import MockCarlaEnv
    
    env = MockCarlaEnv("base_image.png")
    evaluator = YoloEvaluator()
    
    print("Testing Evaluator...")
    
    # 晴天時
    env.set_weather(sun_altitude_angle=90.0, precipitation=0.0, fog_density=0.0)
    img_clear = env.get_image()
    z_clear, conf_clear = evaluator.evaluate(img_clear)
    print(f"Clear Weather -> YOLO3D Z-Distance: {z_clear:.2f}m, Total Conf: {conf_clear:.2f}")
    
    # 悪天候時（暗闇＋霧）
    env.set_weather(sun_altitude_angle=0.0, fog_density=60.0)
    img_bad = env.get_image()
    z_bad, conf_bad = evaluator.evaluate(img_bad)
    print(f"Bad Weather   -> YOLO3D Z-Distance: {z_bad:.2f}m, Total Conf: {conf_bad:.2f}")
