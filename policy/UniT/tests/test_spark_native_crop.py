"""Spark 0908 eval crop must match the training square preformat."""
from __future__ import annotations

import ast
from pathlib import Path


_UNIT_SRC = Path(__file__).resolve().parents[1] / "UniT"
_VIDEO_PY = _UNIT_SRC / "gr00t" / "data" / "transform" / "video.py"
_CONFIG_PY = _UNIT_SRC / "gr00t" / "experiment" / "data_config_unit.py"


def _class_source(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def test_native_crop_class_uses_square_then_256_then_095():
    text = _VIDEO_PY.read_text(encoding="utf-8")
    source = ast.get_source_segment(text, _class_source(_VIDEO_PY, "VideoSparkNativeCrop0908"))
    assert source is not None
    assert "CenterCrop(side)" in source
    assert "Resize(" in source
    assert "(256, 256)" in source
    assert "int(256 * self.scale)" in source


def test_spark_config_uses_native_crop_egovla_does_not():
    text = _CONFIG_PY.read_text(encoding="utf-8")
    spark = ast.get_source_segment(text, _class_source(_CONFIG_PY, "Spark0DualArmDexterousJointDataConfig"))
    threeview = ast.get_source_segment(text, _class_source(_CONFIG_PY, "SparkArenaThreeViewJointDataConfig"))
    ego = ast.get_source_segment(text, _class_source(_CONFIG_PY, "EgoVLAH1Joint38SingleviewDataConfig"))
    assert spark is not None and threeview is not None and ego is not None
    assert "VideoSparkNativeCrop0908(apply_to=self.video_keys, scale=0.95)" in spark
    assert "def transform(" not in threeview
    assert "VideoSparkNativeCrop0908" not in ego
    assert "VideoCrop(apply_to=self.video_keys, scale=0.95)" in ego


def test_eval_crop_sizes_match_the_documented_geometry():
    scale = 0.95
    src_h, src_w = 480, 640
    native_side = min(src_w, src_h)
    native_inner = int(256 * scale)
    plain_h = int(src_h * scale)
    plain_w = int(src_w * scale)
    assert native_side == 480
    assert native_inner == 243
    assert (plain_h, plain_w) == (456, 608)
    train_ref = native_inner / 256 * native_side
    assert abs(train_ref - 455.6) < 0.1
    assert abs(train_ref / plain_w - 0.749) < 0.002
