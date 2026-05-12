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
        画像を推論し、「検出できたターゲット物体の数」と「その信頼度の合計」を返します。
        :param image: BGR形式のnumpy配列画像
        :param return_image: Trueの場合、バウンディングボックス描画済みの画像を一緒に返す
        :return: (検出数, 信頼度の合計) または (検出数, 信頼度の合計, 描画済み画像)
        """
        # YOLOv8による推論
        # verbose=False で標準出力を抑制
        results = self.model(image, verbose=False)
        
        detected_count = 0
        total_confidence = 0.0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 自動運転に関連するクラス（車、人など）のみをカウント
                if cls_id in self.target_classes:
                    detected_count += 1
                    total_confidence += conf
                    
        if return_image:
            # 最初の推論結果（バウンディングボックス等）が描画された画像を取得
            annotated_frame = results[0].plot()
            return detected_count, total_confidence, annotated_frame
            
        return detected_count, total_confidence

if __name__ == "__main__":
    from carla_mock import MockCarlaEnv
    
    env = MockCarlaEnv(r"C:\Users\濱田　紗空\.gemini\antigravity\brain\9a0dbc48-4cc3-47aa-8222-9d812da7d71b\base_dashcam_view_1778238817860.png")
    evaluator = YoloEvaluator()
    
    print("Testing Evaluator...")
    
    # 晴天時
    env.set_weather(sun_altitude_angle=90.0, precipitation=0.0, fog_density=0.0)
    img_clear = env.get_image()
    count_clear, conf_clear = evaluator.evaluate(img_clear)
    print(f"Clear Weather -> Detected: {count_clear}, Total Conf: {conf_clear:.2f}")
    
    # 悪天候時（暗闇＋霧）
    env.set_weather(sun_altitude_angle=0.0, fog_density=60.0)
    img_bad = env.get_image()
    count_bad, conf_bad = evaluator.evaluate(img_bad)
    print(f"Bad Weather   -> Detected: {count_bad}, Total Conf: {conf_bad:.2f}")
