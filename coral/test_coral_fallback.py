"""
Coral(Edge TPU)関連ロジックのセルフチェック（フレームワーク無し、assertベース）。
実機のCoral USB Acceleratorやモデルファイルは無くても実行できる範囲だけを検証する:
  - stress_config.json の "coral" セクションがグローバルへ正しく反映されるか
  - モデル未配置/初期化失敗時に例外を出さずNoneを返し、CPUへフォールバックできるか

実行方法:
    python face_detect/coral/test_coral_fallback.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import peroson_detect as pd  # noqa: E402


def test_load_stress_config_parses_coral_section():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "stress_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "coral": {
                        "enabled": True,
                        "model_path": "custom/model_edgetpu.tflite",
                        "device": ":0",
                    }
                },
                f,
            )

        pd.load_stress_config(cfg_path)
        try:
            assert pd.CORAL_ENABLED is True
            # 相対パスは設定ファイルの場所ではなく OUTPUT_DIR（スクリプト基準）で解決される
            assert pd.CORAL_MODEL_PATH == os.path.normpath(
                os.path.join(pd.OUTPUT_DIR, "custom", "model_edgetpu.tflite")
            )
            assert pd.CORAL_DEVICE == ":0"
        finally:
            # 他のテスト/後続実行に影響しないよう既定値へ戻す
            pd.CORAL_ENABLED = False
            pd.CORAL_DEVICE = ""
    print("OK: load_stress_config が coral セクションを正しく反映する")


def test_missing_model_file_falls_back_to_none():
    result = pd.try_create_coral_emotion_recognizer(
        "does/not/exist_edgetpu.tflite", "", "enet_b2_8"
    )
    assert result is None
    print("OK: モデル未配置時にNoneを返す（例外を出さない）")


def test_invalid_model_file_falls_back_to_none():
    with tempfile.TemporaryDirectory() as tmp:
        bogus_path = os.path.join(tmp, "bogus_edgetpu.tflite")
        with open(bogus_path, "wb") as f:
            f.write(b"not a real tflite model")
        result = pd.try_create_coral_emotion_recognizer(bogus_path, "", "enet_b2_8")
        assert result is None
    print("OK: 不正なモデルファイル/デリゲート未導入時にNoneを返す（例外を出さない）")


def test_idx_to_class_and_img_size_rules():
    assert pd._hsemotion_img_size("enet_b0_8_best_vgaf") == 224
    assert pd._hsemotion_img_size("enet_b2_8") == 260
    assert len(pd._hsemotion_idx_to_class("enet_b2_7")) == 7
    assert len(pd._hsemotion_idx_to_class("enet_b2_8")) == 8
    print("OK: idx_to_class/img_size が hsemotion_onnx と同じ規則で決まる")


if __name__ == "__main__":
    test_load_stress_config_parses_coral_section()
    test_missing_model_file_falls_back_to_none()
    test_invalid_model_file_falls_back_to_none()
    test_idx_to_class_and_img_size_rules()
    print("ALL OK")
