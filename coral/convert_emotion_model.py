"""
HSEmotion(onnx)モデルを Coral USB Accelerator (Edge TPU) 用の .tflite へ変換するスクリプト。

★ Linux(x86-64) 専用 ★
edgetpu_compiler が Linux(x86-64) でしか動かないため、このスクリプトは WSL / Docker / Linux上で
一度だけ実行する（Windows上のperoson_detect.py本体はここで生成した.tfliteを読み込むだけ）。

事前準備（Ubuntu/WSLの例）:
    pip install onnx onnx2tf onnxruntime tensorflow numpy opencv-python-headless
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
        sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update && sudo apt-get install edgetpu-compiler

使い方:
    python convert_emotion_model.py --model-name enet_b2_8 --calib-dir ./calib_faces
    （--calib-dir省略時はランダム画像で代用するが、精度のため実際の顔クロップ画像を
      20〜100枚程度そのフォルダに置くことを推奨する）

出力:
    ./models/<model-name>_edgetpu.tflite
    （peroson_detect.py の stress_config.json "coral.model_path" のデフォルトと同じ場所）
"""

import argparse
import glob
import os
import subprocess
import sys
import urllib.request

import cv2
import numpy as np
import onnx2tf

HSEMOTION_ONNX_URL_TMPL = (
    "https://github.com/HSE-asavchenko/face-emotion-recognition/blob/main/"
    "models/affectnet_emotions/onnx/{model}.onnx?raw=true"
)


def img_size_for(model_name):
    """hsemotion_onnx.HSEmotionRecognizer と同じ入力解像度の規則。"""
    return 224 if "_b0_" in model_name else 260


def preprocess(img, size):
    """本体(peroson_detect.py CoralEmotionRecognizer._preprocess)と同じ前処理。
    量子化キャリブレーションの活性化レンジを実推論と一致させるために揃える。"""
    x = cv2.resize(img, (size, size)) / 255
    x[..., 0] = (x[..., 0] - 0.485) / 0.229
    x[..., 1] = (x[..., 1] - 0.456) / 0.224
    x[..., 2] = (x[..., 2] - 0.406) / 0.225
    return x.astype("float32")[np.newaxis, ...]


def download_onnx(model_name, out_path):
    if os.path.exists(out_path):
        return
    url = HSEMOTION_ONNX_URL_TMPL.format(model=model_name)
    print(f"onnxモデルをダウンロードしています: {url}")
    urllib.request.urlretrieve(url, out_path)


def build_calibration_npy(calib_dir, size, out_path, num_samples=100):
    """代表データセットを1本の .npy (N,H,W,3 float32, 前処理済み) にまとめて保存する。
    onnx2tf の custom_input_op_name_np_data_path はこの形式のファイルを受け取る。
    calib_dir に画像が無ければランダム画像で代用する（量子化はできるが精度は劣化しうる）。"""
    paths = []
    if calib_dir:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(os.path.join(calib_dir, ext)))

    samples = []
    if not paths:
        print(
            "警告: --calib-dir が無いか画像が見つからないため、ランダム画像で量子化します。"
            "精度のため実際の顔クロップ画像の使用を推奨します。"
        )
        rng = np.random.default_rng(0)
        for _ in range(num_samples):
            img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
            samples.append(preprocess(img, size)[0])
    else:
        for p in paths[:num_samples]:
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            samples.append(preprocess(img, size)[0])

    np.save(out_path, np.stack(samples).astype("float32"))
    return out_path


def convert(model_name, calib_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    size = img_size_for(model_name)

    onnx_path = os.path.join(out_dir, f"{model_name}.onnx")
    download_onnx(model_name, onnx_path)

    print("量子化キャリブレーション用データを準備しています...")
    calib_npy = os.path.join(out_dir, f"{model_name}_calib.npy")
    build_calibration_npy(calib_dir, size, calib_npy)

    print("onnx2tf でtflite(int8フル量子化)へ変換しています...")
    # hsemotion_onnx の onnx グラフの入力ノード名は "input"（facial_emotions.py 参照）。
    # mean/std は None にして二重正規化を避ける（正規化は preprocess() 側で済ませてある）。
    # ※ onnx2tf のバージョンによって引数仕様が変わることがある。エラーが出た場合は
    #   実行環境の `onnx2tf --help` / 公式READMEを見てこの呼び出しを調整すること。
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        output_integer_quantized_tflite=True,
        custom_input_op_name_np_data_path=[["input", calib_npy, None, None]],
        verbosity="error",
    )

    quant_tflite = os.path.join(out_dir, f"{model_name}_integer_quant.tflite")
    if not os.path.exists(quant_tflite):
        candidates = glob.glob(os.path.join(out_dir, "*integer_quant*.tflite"))
        if not candidates:
            print("量子化済みtfliteが見つかりません。onnx2tfの出力ファイル名を確認してください。")
            sys.exit(1)
        quant_tflite = candidates[0]

    print(f"edgetpu_compiler を実行しています: {quant_tflite}")
    try:
        subprocess.run(
            ["edgetpu_compiler", "-o", out_dir, quant_tflite],
            check=True,
        )
    except FileNotFoundError:
        print(
            "edgetpu_compiler が見つかりません。Linux(x86-64)上で"
            "Coral公式aptリポジトリからインストールしてください"
            "（このスクリプト冒頭のコメント参照）。"
        )
        sys.exit(1)

    compiled = quant_tflite.replace(".tflite", "_edgetpu.tflite")
    final_name = f"{model_name}_edgetpu.tflite"
    final_path = os.path.join(out_dir, final_name)
    if compiled != final_path and os.path.exists(compiled):
        os.replace(compiled, final_path)
    print(f"完了: {final_path}")
    print(
        "この .tflite を stress_config.json の coral.model_path "
        "（既定: coral/models/emotion_enet_b2_8_edgetpu.tflite）に配置/一致させてください。"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="enet_b2_8")
    parser.add_argument("--calib-dir", default=None, help="顔クロップ画像フォルダ（量子化キャリブレーション用）")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "models"))
    args = parser.parse_args()
    convert(args.model_name, args.calib_dir, args.out)


if __name__ == "__main__":
    main()
