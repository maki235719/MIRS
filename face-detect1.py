"""
カメラ映像からリアルタイムで表情（感情）を認識するスクリプト

必要ライブラリ:
    pip install opencv-python deepface tf-keras

使い方:
    python emotion_recognition.py
    ウィンドウ上で 'q' キーを押すと終了します。
"""

import cv2
from deepface import DeepFace

# DeepFace が返す感情ラベル → 日本語表示
EMOTION_JP = {
    "angry":    "怒り",
    "disgust":  "嫌悪",
    "fear":     "恐怖",
    "happy":    "喜び",
    "sad":      "悲しみ",
    "surprise": "驚き",
    "neutral":  "無表情",
}

# 何フレームごとに解析するか（毎フレーム解析は重いので間引く）
ANALYZE_EVERY = 5


def main():
    cap = cv2.VideoCapture(0)  # 0 = 既定のカメラ
    if not cap.isOpened():
        print("カメラを開けませんでした。デバイス番号や接続を確認してください。")
        return

    frame_count = 0
    last_results = []  # 直近の解析結果（顔ごとの位置と感情）を保持

    print("起動しました。ウィンドウ上で 'q' を押すと終了します。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした。")
            break

        frame_count += 1

        # 一定フレームごとにのみ感情解析を実行
        if frame_count % ANALYZE_EVERY == 0:
            try:
                # enforce_detection=False で顔未検出時も落ちないようにする
                analysis = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                )
                # 複数顔にも対応（analyze は dict または list を返す）
                if isinstance(analysis, dict):
                    analysis = [analysis]
                last_results = analysis
            except Exception as e:
                print(f"解析エラー: {e}")
                last_results = []

        # 解析結果を描画
        for face in last_results:
            region = face.get("region", {})
            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("w", 0)
            h = region.get("h", 0)

            emotion_en = face.get("dominant_emotion", "")
            confidence = face.get("emotion", {}).get(emotion_en, 0.0)
            label = f"{EMOTION_JP.get(emotion_en, emotion_en)} ({confidence:.0f}%)"

            # 顔の枠
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # ラベル背景
            cv2.rectangle(frame, (x, y - 25), (x + max(w, 120), y), (0, 255, 0), -1)
            # ラベル文字（日本語はフォント非対応のため英語表記で描画）
            cv2.putText(
                frame,
                f"{emotion_en} {confidence:.0f}%",
                (x + 3, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        cv2.imshow("Emotion Recognition (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()