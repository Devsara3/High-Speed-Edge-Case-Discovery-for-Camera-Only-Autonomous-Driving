import cv2
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

    def evaluate(self, image, return_image=False):
        """
        画像を推論し、YOLO3Dモデルの出力をエミュレートします。
        ターゲット物体の奥行き（Z成分）を主観的距離として返します。
        
        :param image: BGR形式のnumpy配列画像
        :param return_image: Trueの場合、バウンディングボックス描画済みの画像を一緒に返す
        :return: (最小のZ距離, 信頼度の合計) または (最小のZ距離, 信頼度の合計, 描画済み画像)
                 検出失敗時は Z距離は float('inf') となります。
        """
        results = self.model(image, verbose=False)
        
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
                    
                    # --- YOLO3D 出力のエミュレート部分 ---
                    # 実際のYOLO3Dモデルであれば、ここで box.z のような3D出力が直接取得できます。
                    # 今回は標準のYOLOv8（2D）のため、バウンディングボックスの「ピクセル幅」から
                    # Z成分（奥行き）を近似計算してエミュレートします。
                    
                    # box.xywh[0] は [x_center, y_center, width, height]
                    w_pixel = float(box.xywh[0][2]) 
                    
                    # ピンホールカメラモデルによる簡易的なZ算出
                    # Z = (focal_length * real_world_width) / w_pixel
                    # 仮の定数: focal_length = 800, 車の幅 = 1.8m
                    focal_length = 800.0
                    real_width = 1.8
                    
                    # ピクセル幅が極端に小さい場合のエラー回避
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
