"""
カメラ映像からリアルタイムで表情（感情）を認識し、ストレス度を評価するスクリプト

必要ライブラリ:
    pip install opencv-python deepface tf-keras matplotlib

使い方:
    python emotion_recognition.py
    ウィンドウ上で 'q' キーを押すと終了します（Ctrl+Cでも終了可）。
    終了時に、セッション中のストレス度の推移を stress_log.csv と
    stress_graph.png としてスクリプトと同じフォルダに保存します。
"""

import csv
import os
import time

import cv2
from deepface import DeepFace

import matplotlib
matplotlib.use("Agg")  # imshowウィンドウと競合しない保存専用バックエンド
import matplotlib.pyplot as plt

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

# ストレス寄与の重み（プラスほどストレス増、マイナスほど緩和方向）
# angry/fear/disgust/sad はストレス増、surpriseは軽度に増、
# happy/neutralは緩和（マイナス寄与）として扱う。
STRESS_WEIGHTS = {
    "angry":    1.0,
    "disgust":  0.8,
    "fear":     1.0,
    "sad":      0.7,
    "surprise": 0.3,
    "happy":    -1.0,
    "neutral":  -0.2,
}

# 表示中のスコア変動を滑らかにするための指数移動平均係数（0-1、大きいほど反応が速い）
STRESS_EMA_ALPHA = 0.3

# ストレスバーの色分けしきい値（0-100）
STRESS_LOW_THRESHOLD = 33   # これ未満は緑（低ストレス）
STRESS_HIGH_THRESHOLD = 66  # これ以上は赤（高ストレス）、間は黄

# 出力ファイル（スクリプトと同じフォルダに保存）
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
STRESS_CSV_PATH = os.path.join(OUTPUT_DIR, "stress_log.csv")
STRESS_GRAPH_PATH = os.path.join(OUTPUT_DIR, "stress_graph.png")


def calc_stress_score(emotion_dict):
    """DeepFaceのemotion確率辞書（0-100、合計約100）からストレススコア(0-100)を算出する"""
    raw = sum(
        weight * emotion_dict.get(label, 0.0)
        for label, weight in STRESS_WEIGHTS.items()
    )
    # rawの理論的なレンジはおよそ-100〜+100（全部happyなら-100付近、全部angryなら+100付近）
    # 0-100に正規化してクリップする
    score = (raw + 100.0) / 2.0
    return max(0.0, min(100.0, score))


def stress_color(score):
    """ストレススコアに応じた枠/バーの色 (BGR) を返す（緑→黄→赤）"""
    if score < STRESS_LOW_THRESHOLD:
        return (0, 200, 0)      # 緑
    elif score < STRESS_HIGH_THRESHOLD:
        return (0, 200, 255)    # 黄
    else:
        return (0, 0, 255)      # 赤


def save_stress_report(stress_history):
    """セッション終了時にストレススコアの時系列をCSVとグラフ画像に保存する"""
    if not stress_history:
        print("記録されたストレスデータがないため、ログ・グラフは保存しません。")
        return

    with open(STRESS_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_sec", "frame_count", "stress_score"])
        for elapsed, frame_count, score in stress_history:
            writer.writerow([f"{elapsed:.2f}", frame_count, f"{score:.1f}"])
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


def main():
    cap = cv2.VideoCapture(0)  # 0 = 既定のカメラ
    if not cap.isOpened():
        print("カメラを開けませんでした。デバイス番号や接続を確認してください。")
        return

    frame_count = 0
    last_results = []  # 直近の解析結果（顔ごとの位置と感情）を保持

    stress_history = []  # (経過秒, フレーム番号, スコア) のタプルのリスト
    session_start = time.time()
    smoothed_stress = None  # 主要顔のセッション全体ストレス（EMA平滑化後）

    print("起動しました。ウィンドウ上で 'q' を押すと終了します。")

    try:
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

                # 複数顔の中から最大の顔を「主要顔」とし、セッション全体の
                # ストレス推移（EMA・CSV・グラフ）に使う。顔が無い回は
                # 直前の値を保持し、記録もスキップする（ジッター防止）。
                if last_results:
                    primary_face = max(
                        last_results,
                        key=lambda f: f.get("region", {}).get("w", 0)
                        * f.get("region", {}).get("h", 0),
                    )
                    raw_score = calc_stress_score(primary_face.get("emotion", {}))
                    if smoothed_stress is None:
                        smoothed_stress = raw_score
                    else:
                        smoothed_stress = (
                            STRESS_EMA_ALPHA * raw_score
                            + (1 - STRESS_EMA_ALPHA) * smoothed_stress
                        )
                    elapsed = time.time() - session_start
                    stress_history.append((elapsed, frame_count, smoothed_stress))

            # 解析結果を描画
            for face in last_results:
                region = face.get("region", {})
                x = region.get("x", 0)
                y = region.get("y", 0)
                w = region.get("w", 0)
                h = region.get("h", 0)

                emotion_en = face.get("dominant_emotion", "")
                confidence = face.get("emotion", {}).get(emotion_en, 0.0)
                face_stress = calc_stress_score(face.get("emotion", {}))

                # 顔の枠
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # ラベル背景
                cv2.rectangle(frame, (x, y - 25), (x + max(w, 180), y), (0, 255, 0), -1)
                # ラベル文字（日本語はフォント非対応のため英語表記で描画）
                cv2.putText(
                    frame,
                    f"{emotion_en} {confidence:.0f}% | Stress {face_stress:.0f}",
                    (x + 3, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                )

            # セッション全体のストレスゲージ（画面左上に固定表示）
            gauge_x, gauge_y, gauge_w, gauge_h = 20, 20, 200, 20
            display_score = smoothed_stress if smoothed_stress is not None else 0.0
            gauge_fill_color = stress_color(display_score)
            cv2.rectangle(
                frame,
                (gauge_x, gauge_y),
                (gauge_x + gauge_w, gauge_y + gauge_h),
                (200, 200, 200),
                2,
            )
            fill_width = int(gauge_w * display_score / 100)
            if fill_width > 0:
                cv2.rectangle(
                    frame,
                    (gauge_x, gauge_y),
                    (gauge_x + fill_width, gauge_y + gauge_h),
                    gauge_fill_color,
                    -1,
                )
            cv2.putText(
                frame,
                f"Stress: {display_score:.0f}",
                (gauge_x, gauge_y + gauge_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                gauge_fill_color,
                2,
            )

            cv2.imshow("Stress Evaluation (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Ctrl+Cで中断されました。")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_stress_report(stress_history)


if __name__ == "__main__":
    main()
