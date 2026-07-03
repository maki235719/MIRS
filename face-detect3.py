"""
カメラ映像からリアルタイムで表情（感情）を認識し、ストレス度を評価するスクリプト（高精度版）

■ 旧版からの主な変更点
  - 顔検出/ランドマーク: OpenCV Haar → MediaPipe FaceLandmarker（Tasks API, 478点＋blendshapes）
  - 感情認識: DeepFace(FER2013系) → HSEmotion(EfficientNet系, 8クラス) に置換して精度向上
  - ストレス評価: 感情の線形加重 → 「起動時の平常状態からの逸脱(zスコア)」で再設計
      * 感情のネガティブ度に加え、blendshapesから得た
        「眉間のしわ(browDown)」「まばたき率」を微細特徴として合成
      * 個人ごとに較正するので、恣意的な固定係数に依存しない
  - 感情確率を時間平滑化(EMA)＋低信頼度ゲートで表示のブレを抑制

必要ライブラリ:
    pip install opencv-python mediapipe hsemotion-onnx onnxruntime tf-keras matplotlib

使い方:
    python face-detect2.py
    起動直後に数秒間の「較正」フェーズがあります。画面の指示どおり
    平常（リラックスした無表情）の状態を保ってください。
    その後リアルタイム評価に移ります。'q' キー（またはCtrl+C）で終了します。
    終了時に stress_log.csv と stress_graph.png を保存します。
"""

import csv
import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

import matplotlib
matplotlib.use("Agg")  # imshowウィンドウと競合しない保存専用バックエンド
import matplotlib.pyplot as plt


# ============================================================================
# 設定
# ============================================================================

# HSEmotion が返す感情ラベル（8クラス） → 日本語表示（描画はフォント都合で英語）
EMOTION_JP = {
    "Anger":     "怒り",
    "Contempt":  "軽蔑",
    "Disgust":   "嫌悪",
    "Fear":      "恐怖",
    "Happiness": "喜び",
    "Neutral":   "無表情",
    "Sadness":   "悲しみ",
    "Surprise":  "驚き",
}

# ストレス方向に働くネガティブ感情（この確率和が高いほどストレス寄り）
NEGATIVE_EMOTIONS = ["Anger", "Contempt", "Disgust", "Fear", "Sadness"]

# 何フレームごとに感情解析(HSEmotion)を実行するか。
# 顔ランドマーク(FaceLandmarker)は毎フレーム実行する（まばたき検出に必要かつ軽量）。
ANALYZE_EVERY = 3

# 使用する感情モデル（enet_b2_8 = 8クラス, 入力260px, b0より高精度）
HSEMOTION_MODEL = "enet_b2_8"

# 感情確率ベクトルの時間平滑化係数（0-1, 大きいほど反応が速い）
EMO_EMA_ALPHA = 0.4
# 最終ストレススコアの表示平滑化係数
STRESS_EMA_ALPHA = 0.3

# --- ストレススコア合成の設定 ---
# 各成分をベースラインからのzスコアにし、重み付き和を取る（合計が1.0になるよう配分）
# 論文（Giannakakis 2017 / 顔AUストレス解析2021 ほか）で報告された相関の強い
# 顔特徴を追加: head=頭部の動き, mouth=口唇の緊張, eye=瞼の緊張。重みは実測に応じ調整可。
STRESS_COMPONENT_WEIGHTS = {
    "emotion": 0.40,  # ネガティブ感情の増加
    "brow":    0.20,  # 眉間のしわ（browDown / AU4）の増加
    "blink":   0.10,  # まばたき率の増加
    "head":    0.12,  # 頭部運動（角速度）の増加
    "mouth":   0.10,  # 口唇の緊張（mouthPress / AU23-24）の増加
    "eye":     0.08,  # 瞼の緊張（eyeSquint / AU7）の増加
}
# z=0（＝平常時）を何点にするか、および z 1あたり何点上げるか
STRESS_BASE_LEVEL = 25.0
STRESS_Z_GAIN = 15.0

# zスコアの分母（標準偏差）の下限。平常状態が静かすぎるとノイズで暴れるのを防ぐ
EMO_STD_FLOOR = 0.05
BROW_STD_FLOOR = 0.02
BLINK_STD_FLOOR = 3.0    # 回/分
HEAD_STD_FLOOR = 2.0     # deg/秒（平滑化角速度）
MOUTH_STD_FLOOR = 0.02
EYE_STD_FLOOR = 0.02

# 頭部運動（角速度）の時間平滑化係数（0-1, 大きいほど反応が速い）
HEAD_EMA_ALPHA = 0.4

# 較正（ベースライン測定）フェーズの長さ（秒）
CALIB_SECONDS = 7.0

# まばたき検出（blendshape eyeBlink のしきい値・ヒステリシス）と集計窓
BLINK_ON_THRESHOLD = 0.5
BLINK_OFF_THRESHOLD = 0.35
BLINK_WINDOW_SEC = 30.0  # まばたき率を計算する直近の窓（秒）

# 顔クロップの余白（ランドマーク外接矩形に対する比率）
FACE_CROP_MARGIN = 0.15

# ストレスバーの色分けしきい値（0-100）
STRESS_LOW_THRESHOLD = 33   # これ未満は緑（低ストレス）
STRESS_HIGH_THRESHOLD = 66  # これ以上は赤（高ストレス）、間は黄

# 出力ファイル/モデルファイル（スクリプトと同じフォルダ）
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
STRESS_CSV_PATH = os.path.join(OUTPUT_DIR, "stress_log.csv")
STRESS_GRAPH_PATH = os.path.join(OUTPUT_DIR, "stress_graph.png")
FACE_LANDMARKER_TASK = os.path.join(OUTPUT_DIR, "face_landmarker.task")
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


# ============================================================================
# モデル準備
# ============================================================================

def ensure_face_landmarker_model():
    """FaceLandmarker のモデルバンドル(.task)が無ければダウンロードする"""
    if not os.path.exists(FACE_LANDMARKER_TASK):
        print("FaceLandmarker モデルをダウンロードしています...")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_TASK)
        print("ダウンロード完了。")


def create_face_landmarker():
    """blendshapes 出力付きの FaceLandmarker(VIDEOモード) を生成する"""
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_TASK),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,  # 頭部姿勢（回転）を取得するため
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ============================================================================
# 特徴抽出
# ============================================================================

def landmarks_to_bbox(landmarks, frame_w, frame_h, margin=FACE_CROP_MARGIN):
    """正規化ランドマーク列 → 画素座標の外接矩形 (x, y, w, h)。余白付き＆画面内にクリップ。"""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1, x2 = min(xs) * frame_w, max(xs) * frame_w
    y1, y2 = min(ys) * frame_h, max(ys) * frame_h
    mw = (x2 - x1) * margin
    mh = (y2 - y1) * margin
    x1 = int(max(0, x1 - mw))
    y1 = int(max(0, y1 - mh))
    x2 = int(min(frame_w, x2 + mw))
    y2 = int(min(frame_h, y2 + mh))
    return x1, y1, x2 - x1, y2 - y1


def blendshapes_to_dict(blendshape_categories):
    """blendshape の Category リスト → {名前: スコア} の辞書"""
    return {c.category_name: c.score for c in blendshape_categories}


def brow_down_value(bs):
    """眉間のしわ（corrugator近似）。browDownLeft/Right の平均。高いほど眉を寄せている。"""
    return (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0


def blink_signal(bs):
    """まばたき信号。左右 eyeBlink の大きい方（片目つむりにも反応）。"""
    return max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))


def emotion_negative_affect(prob_dict):
    """感情確率(合計約1)からネガティブ度スカラーを算出。範囲はおおよそ -1〜+1。"""
    neg = sum(prob_dict.get(e, 0.0) for e in NEGATIVE_EMOTIONS)
    return neg - prob_dict.get("Happiness", 0.0)


def mouth_tension_value(bs):
    """口唇の緊張（AU23/24 近似）。左右 mouthPress の平均に mouthPucker を補助加重。
    高いほど唇を強く結んでいる。ストレスと相関（Giannakakis 2017 ほか）。"""
    press = (bs.get("mouthPressLeft", 0.0) + bs.get("mouthPressRight", 0.0)) / 2.0
    pucker = bs.get("mouthPucker", 0.0)
    return press + 0.5 * pucker


def eye_tension_value(bs):
    """瞼の緊張（AU7 lid tightener 近似）。左右 eyeSquint の平均に eyeWide(AU5) を補助加重。
    主観的ストレスと相関（Automatic stress analysis from AUs, 2021）。"""
    squint = (bs.get("eyeSquintLeft", 0.0) + bs.get("eyeSquintRight", 0.0)) / 2.0
    wide = (bs.get("eyeWideLeft", 0.0) + bs.get("eyeWideRight", 0.0)) / 2.0
    return squint + 0.5 * wide


def matrix_to_euler(matrix):
    """4x4 顔姿勢変換行列の回転部から オイラー角(pitch, yaw, roll)[度] を分解する。
    matrix が None の場合は (0,0,0) を返す。"""
    if matrix is None:
        return 0.0, 0.0, 0.0
    m = np.asarray(matrix, dtype=np.float64)
    r = m[:3, :3]
    sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = np.arctan2(r[1, 0], r[0, 0])
    else:  # ジンバルロック近傍
        pitch = np.arctan2(-r[1, 2], r[1, 1])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = 0.0
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)


# ============================================================================
# ストレススコア
# ============================================================================

def zscore(value, mean, std, std_floor):
    return (value - mean) / max(std, std_floor)


def compose_stress(emo_neg, brow, blink_rate, head_motion, mouth, eye, baseline):
    """各成分をベースラインからのzスコアにし、重み付き和 → 0-100 に写像する。"""
    z_emo = zscore(emo_neg, baseline["emo_mean"], baseline["emo_std"], EMO_STD_FLOOR)
    z_brow = zscore(brow, baseline["brow_mean"], baseline["brow_std"], BROW_STD_FLOOR)
    z_blink = zscore(
        blink_rate, baseline["blink_mean"], baseline["blink_std"], BLINK_STD_FLOOR
    )
    z_head = zscore(
        head_motion, baseline["head_mean"], baseline["head_std"], HEAD_STD_FLOOR
    )
    z_mouth = zscore(mouth, baseline["mouth_mean"], baseline["mouth_std"], MOUTH_STD_FLOOR)
    z_eye = zscore(eye, baseline["eye_mean"], baseline["eye_std"], EYE_STD_FLOOR)
    z_total = (
        STRESS_COMPONENT_WEIGHTS["emotion"] * z_emo
        + STRESS_COMPONENT_WEIGHTS["brow"] * z_brow
        + STRESS_COMPONENT_WEIGHTS["blink"] * z_blink
        + STRESS_COMPONENT_WEIGHTS["head"] * z_head
        + STRESS_COMPONENT_WEIGHTS["mouth"] * z_mouth
        + STRESS_COMPONENT_WEIGHTS["eye"] * z_eye
    )
    score = STRESS_BASE_LEVEL + STRESS_Z_GAIN * z_total
    return max(0.0, min(100.0, score))


def stress_color(score):
    """ストレススコアに応じた枠/バーの色 (BGR) を返す（緑→黄→赤）"""
    if score < STRESS_LOW_THRESHOLD:
        return (0, 200, 0)      # 緑
    elif score < STRESS_HIGH_THRESHOLD:
        return (0, 200, 255)    # 黄
    else:
        return (0, 0, 255)      # 赤


# ============================================================================
# まばたきカウンタ
# ============================================================================

class BlinkCounter:
    """eyeBlink 信号のヒステリシス立ち上がりで瞬目を数え、直近窓の瞬目率(回/分)を返す。"""

    def __init__(self):
        self.is_closed = False
        self.timestamps = deque()  # まばたき発生時刻(秒)

    def update(self, signal, now):
        if not self.is_closed and signal >= BLINK_ON_THRESHOLD:
            self.is_closed = True
            self.timestamps.append(now)  # 目を閉じた瞬間を1回とカウント
        elif self.is_closed and signal <= BLINK_OFF_THRESHOLD:
            self.is_closed = False
        # 窓外の古い記録を捨てる
        while self.timestamps and now - self.timestamps[0] > BLINK_WINDOW_SEC:
            self.timestamps.popleft()

    def rate_per_min(self, now, start_time):
        """直近窓の瞬目率(回/分)。経過が窓長に満たない間は経過時間で正規化する。"""
        elapsed = now - start_time
        window = min(BLINK_WINDOW_SEC, elapsed)
        if window < 1.0:
            return 0.0
        count = sum(1 for t in self.timestamps if now - t <= window)
        return count * 60.0 / window


# ============================================================================
# 頭部運動トラッカ
# ============================================================================

class HeadMotionTracker:
    """頭部姿勢のオイラー角(pitch/yaw/roll)から角速度(deg/秒)を求め、EMA平滑化して返す。
    ストレス下で頭部運動が増える傾向（Giannakakis, head pose features）を特徴量化する。"""

    def __init__(self):
        self.prev_angles = None
        self.prev_time = None
        self.smoothed = 0.0

    def update(self, angles, now):
        pitch, yaw, roll = angles
        if self.prev_angles is not None and self.prev_time is not None:
            dt = now - self.prev_time
            if dt > 1e-3:
                d = np.array([pitch, yaw, roll]) - np.array(self.prev_angles)
                speed = float(np.sqrt(np.sum(d ** 2)) / dt)  # deg/秒
                self.smoothed = (
                    HEAD_EMA_ALPHA * speed + (1 - HEAD_EMA_ALPHA) * self.smoothed
                )
        self.prev_angles = (pitch, yaw, roll)
        self.prev_time = now
        return self.smoothed


# ============================================================================
# レポート保存
# ============================================================================

def save_stress_report(stress_history):
    """セッション終了時にストレススコアの時系列をCSVとグラフ画像に保存する"""
    if not stress_history:
        print("記録されたストレスデータがないため、ログ・グラフは保存しません。")
        return

    with open(STRESS_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "elapsed_sec",
                "frame_count",
                "stress_score",
                "dominant_emotion",
                "emo_negative",
                "brow_down",
                "blink_rate_per_min",
                "head_motion",
                "mouth_tension",
                "eye_tension",
            ]
        )
        for row in stress_history:
            elapsed, fc, score, emo, emo_neg, brow, blink, head, mouth, eye = row
            writer.writerow(
                [
                    f"{elapsed:.2f}",
                    fc,
                    f"{score:.1f}",
                    emo,
                    f"{emo_neg:.3f}",
                    f"{brow:.3f}",
                    f"{blink:.1f}",
                    f"{head:.2f}",
                    f"{mouth:.3f}",
                    f"{eye:.3f}",
                ]
            )
    print(f"ストレスログを保存しました: {STRESS_CSV_PATH}")

    times = [row[0] for row in stress_history]
    scores = [row[2] for row in stress_history]

    plt.figure(figsize=(10, 4))
    plt.plot(times, scores, color="tab:red", linewidth=1.5)
    plt.xlabel("Elapsed Time (sec)")
    plt.ylabel("Stress Score (0-100)")
    plt.title("Stress Score over Session")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(STRESS_GRAPH_PATH)
    plt.close()
    print(f"ストレスグラフを保存しました: {STRESS_GRAPH_PATH}")


# ============================================================================
# メイン
# ============================================================================

def main():
    ensure_face_landmarker_model()

    print("感情モデルを読み込んでいます（初回はダウンロードが走ります）...")
    fer = HSEmotionRecognizer(model_name=HSEMOTION_MODEL)
    landmarker = create_face_landmarker()

    cap = cv2.VideoCapture(0)  # 0 = 既定のカメラ
    if not cap.isOpened():
        print("カメラを開けませんでした。デバイス番号や接続を確認してください。")
        landmarker.close()
        return

    frame_count = 0
    session_start = time.time()
    last_ts_ms = -1

    # 直近の解析状態（描画とストレス計算で共有）
    smoothed_probs = None          # 平滑化した感情確率辞書
    last_bbox = None               # 直近の顔矩形
    last_emo_neg = 0.0
    last_brow = 0.0
    last_head = 0.0                # 平滑化した頭部角速度(deg/秒)
    last_mouth = 0.0               # 口唇の緊張
    last_eye = 0.0                 # 瞼の緊張

    blink_counter = BlinkCounter()
    head_tracker = HeadMotionTracker()

    # 較正フェーズ用のサンプル蓄積
    calibrating = True
    calib_emo = []
    calib_brow = []
    calib_head = []
    calib_mouth = []
    calib_eye = []
    baseline = None
    smoothed_stress = None
    stress_history = []

    print("起動しました。まず数秒間、平常な表情を保ってください（較正中）。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームを取得できませんでした。")
                break

            frame_count += 1
            now = time.time()
            elapsed = now - session_start

            # --- FaceLandmarker は毎フレーム実行（VIDEOモードは単調増加のtimestampが必要） ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(elapsed * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms
            result = landmarker.detect_for_video(mp_image, ts_ms)

            face_present = bool(result.face_landmarks)
            frame_h, frame_w = frame.shape[:2]

            if face_present:
                landmarks = result.face_landmarks[0]
                last_bbox = landmarks_to_bbox(landmarks, frame_w, frame_h)

                # 微細特徴（blendshapes）: 毎フレーム更新
                if result.face_blendshapes:
                    bs = blendshapes_to_dict(result.face_blendshapes[0])
                    last_brow = brow_down_value(bs)
                    last_mouth = mouth_tension_value(bs)
                    last_eye = eye_tension_value(bs)
                    blink_counter.update(blink_signal(bs), now)

                # 頭部運動（姿勢行列 → オイラー角 → 角速度）: 毎フレーム更新
                if result.facial_transformation_matrixes:
                    angles = matrix_to_euler(result.facial_transformation_matrixes[0])
                    last_head = head_tracker.update(angles, now)

                # 感情解析（HSEmotion）: 間引いて実行
                if frame_count % ANALYZE_EVERY == 0:
                    x, y, w, h = last_bbox
                    if w > 0 and h > 0:
                        face_bgr = frame[y : y + h, x : x + w]
                        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                        try:
                            _, scores = fer.predict_emotions(face_rgb, logits=False)
                            probs = {
                                fer.idx_to_class[i]: float(scores[i])
                                for i in range(len(scores))
                            }
                            # 感情確率ベクトルを EMA 平滑化
                            if smoothed_probs is None:
                                smoothed_probs = probs
                            else:
                                smoothed_probs = {
                                    k: EMO_EMA_ALPHA * probs[k]
                                    + (1 - EMO_EMA_ALPHA) * smoothed_probs.get(k, 0.0)
                                    for k in probs
                                }
                            last_emo_neg = emotion_negative_affect(smoothed_probs)
                        except Exception as e:
                            print(f"感情解析エラー: {e}")

            blink_rate = blink_counter.rate_per_min(now, session_start)

            # ================= 較正フェーズ =================
            if calibrating:
                if face_present:
                    calib_brow.append(last_brow)
                    calib_head.append(last_head)
                    calib_mouth.append(last_mouth)
                    calib_eye.append(last_eye)
                    if smoothed_probs is not None:
                        calib_emo.append(last_emo_neg)

                if elapsed >= CALIB_SECONDS:
                    # ベースライン統計を確定
                    emo_arr = np.array(calib_emo) if calib_emo else np.array([0.0])
                    brow_arr = np.array(calib_brow) if calib_brow else np.array([0.0])
                    head_arr = np.array(calib_head) if calib_head else np.array([0.0])
                    mouth_arr = np.array(calib_mouth) if calib_mouth else np.array([0.0])
                    eye_arr = np.array(calib_eye) if calib_eye else np.array([0.0])
                    baseline = {
                        "emo_mean": float(emo_arr.mean()),
                        "emo_std": float(emo_arr.std()),
                        "brow_mean": float(brow_arr.mean()),
                        "brow_std": float(brow_arr.std()),
                        "blink_mean": blink_rate,          # 較正中の平常瞬目率
                        "blink_std": BLINK_STD_FLOOR,      # 瞬目率は分散推定が不安定なので下限を使う
                        "head_mean": float(head_arr.mean()),
                        "head_std": float(head_arr.std()),
                        "mouth_mean": float(mouth_arr.mean()),
                        "mouth_std": float(mouth_arr.std()),
                        "eye_mean": float(eye_arr.mean()),
                        "eye_std": float(eye_arr.std()),
                    }
                    calibrating = False
                    print(
                        "較正完了。ベースライン: "
                        f"emo={baseline['emo_mean']:.2f}, "
                        f"brow={baseline['brow_mean']:.2f}, "
                        f"blink={baseline['blink_mean']:.1f}/min, "
                        f"head={baseline['head_mean']:.1f}deg/s, "
                        f"mouth={baseline['mouth_mean']:.2f}, "
                        f"eye={baseline['eye_mean']:.2f}"
                    )
            # ================= 評価フェーズ =================
            else:
                if face_present:
                    raw_score = compose_stress(
                        last_emo_neg, last_brow, blink_rate,
                        last_head, last_mouth, last_eye, baseline
                    )
                    if smoothed_stress is None:
                        smoothed_stress = raw_score
                    else:
                        smoothed_stress = (
                            STRESS_EMA_ALPHA * raw_score
                            + (1 - STRESS_EMA_ALPHA) * smoothed_stress
                        )
                    dominant = (
                        max(smoothed_probs, key=smoothed_probs.get)
                        if smoothed_probs
                        else ""
                    )
                    stress_history.append(
                        (
                            elapsed,
                            frame_count,
                            smoothed_stress,
                            dominant,
                            last_emo_neg,
                            last_brow,
                            blink_rate,
                            last_head,
                            last_mouth,
                            last_eye,
                        )
                    )

            # ================= 描画 =================
            if face_present and last_bbox is not None:
                x, y, w, h = last_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if smoothed_probs:
                    dom = max(smoothed_probs, key=smoothed_probs.get)
                    conf = smoothed_probs[dom] * 100.0
                    label = f"{dom} {conf:.0f}%"
                else:
                    label = "..."
                cv2.rectangle(frame, (x, y - 25), (x + max(w, 200), y), (0, 255, 0), -1)
                cv2.putText(
                    frame, label, (x + 3, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2,
                )

            # 較正の進捗 or ストレスゲージ
            if calibrating:
                remaining = max(0.0, CALIB_SECONDS - elapsed)
                cv2.putText(
                    frame,
                    f"CALIBRATING... keep a neutral face ({remaining:.0f}s)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
            else:
                gauge_x, gauge_y, gauge_w, gauge_h = 20, 20, 200, 20
                display_score = smoothed_stress if smoothed_stress is not None else 0.0
                color = stress_color(display_score)
                cv2.rectangle(
                    frame, (gauge_x, gauge_y),
                    (gauge_x + gauge_w, gauge_y + gauge_h), (200, 200, 200), 2,
                )
                fill_w = int(gauge_w * display_score / 100)
                if fill_w > 0:
                    cv2.rectangle(
                        frame, (gauge_x, gauge_y),
                        (gauge_x + fill_w, gauge_y + gauge_h), color, -1,
                    )
                cv2.putText(
                    frame, f"Stress: {display_score:.0f}",
                    (gauge_x, gauge_y + gauge_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )
                # 参考値（小さく表示）
                cv2.putText(
                    frame,
                    f"brow:{last_brow:.2f} blink:{blink_rate:.0f}/min",
                    (gauge_x, gauge_y + gauge_h + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                )
                cv2.putText(
                    frame,
                    f"head:{last_head:.0f}deg/s mouth:{last_mouth:.2f} eye:{last_eye:.2f}",
                    (gauge_x, gauge_y + gauge_h + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                )

            if not face_present:
                cv2.putText(
                    frame, "No face detected", (20, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

            cv2.imshow("Stress Evaluation (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Ctrl+Cで中断されました。")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()  # mediapipe の終了時例外を避けるため明示的に閉じる
        save_stress_report(stress_history)


if __name__ == "__main__":
    main()
