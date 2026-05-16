"""
1-C. 深度推定（Depth Estimation）の実験
カメラ画像から距離を測定するための3つのアプローチの推論パイプライン。
"""

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python (cv2) is not installed.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch is not installed.")


class MiDaS_DepthEstimator:
    """
    MiDaS: 自動車以外にも汎用的に使える、混合データセットで訓練された深度推定モデル
    """
    def __init__(self, model_type="DPT_Hybrid", device=None):
        if not TORCH_AVAILABLE:
            self.midas = None
            return
            
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading MiDaS model ({model_type}) on {self.device}...")
        
        # PyTorch HubからMiDaSをロード (DPT_Large, DPT_Hybrid, MiDaS_small などが選択可能)
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval()
        
        # モデルに合わせた前処理トランスフォームをロード
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def estimate(self, image_rgb):
        """
        RGB画像を入力とし、相対的な深度マップ(Relative Depth)を出力する
        """
        if self.midas is None:
            print("Running MiDaS in dummy mode (torch not available)")
            return np.ones(image_rgb.shape[:2], dtype=np.float32)

        input_batch = self.transform(image_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # 元の画像サイズにリサイズ
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Numpy配列に変換
        depth_map = prediction.cpu().numpy()
        return depth_map


class StereoGeometryEstimator:
    """
    Stereo（複眼）ジオメトリ: 2つのカメラシステムを使い、幾何学から深度 d を計算する数理モデル
    OpenCVの StereoSGBM (Semi-Global Block Matching) を使用。
    """
    def __init__(self, focal_length, baseline):
        """
        focal_length: カメラの焦点距離 (ピクセル単位)
        baseline: 左右のカメラの間隔 (メートル)
        """
        self.focal_length = focal_length
        self.baseline = baseline
        self.stereo = None
        
        if CV2_AVAILABLE:
            # SGBMアルゴリズムの初期化
            window_size = 5
            min_disp = 0
            num_disp = 16 * 5 # 16の倍数である必要がある
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=window_size,
                P1=8 * 3 * window_size**2,
                P2=32 * 3 * window_size**2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )

    def estimate(self, image_left, image_right):
        """
        左右のステレオ画像を入力し、実スケールの深度マップ(メートル)を計算する。
        """
        if self.stereo is None:
            print("Running Stereo in dummy mode (cv2 not available)")
            return np.ones(image_left.shape[:2], dtype=np.float32)

        # グレースケールに変換
        gray_left = cv2.cvtColor(image_left, cv2.COLOR_RGB2GRAY)
        gray_right = cv2.cvtColor(image_right, cv2.COLOR_RGB2GRAY)
        
        # 視差（Disparity）の計算
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # ゼロ割りを防ぐ
        disparity[disparity <= 0] = 0.1
        
        # 深度計算: Depth = (focal_length * baseline) / disparity
        depth_map = (self.focal_length * self.baseline) / disparity
        
        return depth_map


class MonoDepth2Estimator:
    """
    MonoDepth2: 自己監視型の単眼深度推定システム
    ※ 車両の移動（ビデオシーケンス）やステレオ画像を使って自己監視学習を行うモデル。
    推論時は単一の画像から深度を予測する。
    """
    def __init__(self, weights_path=None, device='cpu'):
        self.device = device
        print("Initializing MonoDepth2 inference pipeline...")
        # 実際にはここに ResNet バックボーンと Depth Decoder のネットワークが構築されます。
        # self.encoder = ResnetEncoder(...)
        # self.decoder = DepthDecoder(...)
        # if weights_path:
        #     load_weights(...)

    def estimate(self, image_rgb):
        """
        単眼画像からの推論パイプライン。
        """
        # ダミー処理: 実際の推論ではエンコーダとデコーダを通す
        height, width = image_rgb.shape[:2]
        dummy_depth = np.ones((height, width), dtype=np.float32) * 10.0 # 10mのダミー深度
        return dummy_depth


if __name__ == "__main__":
    # 動作確認用モック
    print("--- 動作確認モック ---")
    dummy_img_l = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_img_r = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 1. MiDaS (PCにPyTorchがあれば実際にダウンロードして動きます。重いのでデフォルトではコメントアウト)
    # midas = MiDaS_DepthEstimator(model_type="MiDaS_small")
    # depth_midas = midas.estimate(dummy_img_l)
    # print("MiDaS output shape:", depth_midas.shape)
    
    # 2. Stereo
    stereo = StereoGeometryEstimator(focal_length=700.0, baseline=0.54)
    depth_stereo = stereo.estimate(dummy_img_l, dummy_img_r)
    print("Stereo Depth output shape:", depth_stereo.shape)
    
    # 3. MonoDepth2
    mono = MonoDepth2Estimator()
    depth_mono = mono.estimate(dummy_img_l)
    print("MonoDepth2 output shape:", depth_mono.shape)
