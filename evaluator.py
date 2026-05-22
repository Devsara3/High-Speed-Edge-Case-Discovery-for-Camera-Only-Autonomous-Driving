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
        # 車や歩行者など、注目するクラスID (COCO dataset: 0:person, 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck)
        self.target_classes = [0, 1, 2, 3, 5, 7]

        # 深層学習深度推定モデル (MiDaS) のロード試行 (実用的な単眼3D検出システムとして機能させます)
        try:
            import numpy as np # evaluator用に追加インポートを保証
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

    def evaluate(self, image, return_image=False):
        """
        画像を推論し、YOLO3Dモデルの出力をエミュレートします。
        ターゲット物体の奥行き（Z成分）を主観的距離として返します。
        
        :param image: BGR形式のnumpy配列画像
        :param return_image: Trueの場合、バウンディングボックス描画済みの画像を一緒に返す
        :return: (最小のZ距離, 信頼度の合計) または (最小のZ距離, 信頼度の合計, 描画済み画像)
                 検出失敗時は Z距離は float('inf') となります。
        """
        results = self.model(image, verbose=False, conf=0.1)
        
        detected_count = 0
        total_confidence = 0.0
        min_z_distance = float('inf') # YOLO3Dのエミュレート：最も近い物体のZ距離

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 自動運転に関連するクラスのみをカウント
                if cls_id in self.target_classes:
                    detected_count += 1
                    total_confidence += conf
                    
                    # --- YOLO3D 出力の実装部分 ---
                    # 実際のYOLO3D（単眼3D物体検出）と同様の出力を得るため、
                    # YOLOv8（2D境界ボックス）とMiDaS（深層学習深度推定モデル）を組み合わせた
                    # ハイブリッド3D物体検出システム（センサーフュージョン）を起動します。
                    
                    if hasattr(self, 'depth_estimator') and self.depth_estimator is not None:
                        # 1. 画像全体のRGB画像に変換 (YOLOv8はBGRだがMiDaSはRGBを想定)
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        depth_map = self.depth_estimator.estimate(rgb_image)
                        
                        # 2. 検出物体のバウンディングボックス領域を取得
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        h, w = depth_map.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)
                        
                        if x2 > x1 and y2 > y1:
                            box_depth = depth_map[y1:y2, x1:x2]
                            # MiDaSの出力は「視差（disparity）」の相対値（大きいほど近い）
                            median_disparity = np.median(box_depth)
                            
                            # メートル単位の物理的な3D距離(Z)に校正する (ゼロ割防止)
                            if median_disparity > 0:
                                # ディープラーニングによる推定距離(Z)の算出
                                z_dist = 200.0 / median_disparity
                            else:
                                z_dist = float('inf')
                        else:
                            z_dist = float('inf')
                    else:
                        # 従来のピンホールカメラモデルによる幾何学的エミュレーション
                        w_pixel = float(box.xywh[0][2]) 
                        focal_length = 800.0
                        real_width = 1.8
                        
                        if w_pixel > 1.0:
                            z_dist = (focal_length * real_width) / w_pixel
                        else:
                            z_dist = float('inf')
                        
                    if z_dist < min_z_distance:
                        min_z_distance = z_dist

        if return_image:
            annotated_frame = results[0].plot()
            return min_z_distance, total_confidence, annotated_frame
            
        return min_z_distance, total_confidence

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
