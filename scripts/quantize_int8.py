"""
quantize_int8.py
----------------
Builds an INT8 (static, post-training) version of MobileFaceNet for the edge device.

    python scripts/quantize_int8.py
    -> uno_q/models/w600k_mbf_int8.onnx

Steps: opset 11 -> 13, ONNX Runtime pre-processing, static quantization (QOperator,
UINT8 activations / INT8 weights) calibrated on the aligned dataset crops. The final
Flatten -> Gemm -> BatchNorm embedding head stays FP32 because quantizing it costs the
most accuracy.

IMPORTANT: INT8 and FP32 embeddings are NOT interchangeable (cosine ~0.89 between them on
our data). If the device runs the INT8 model (REC_MODEL_PATH), every customer must be
enrolled with the INT8 model too. Measure on the UNO Q before choosing:
    python scripts/evaluate.py --model uno_q/models/w600k_mbf_int8.onnx
    python scripts/benchmark_edge.py --int8 uno_q/models/w600k_mbf_int8.onnx
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import version_converter
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quant_pre_process, quantize_static)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "uno_q"))
import facerec_uno_q as fr  # noqa: E402


def preprocess(img: np.ndarray) -> np.ndarray:
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return ((x - 127.5) / 127.5).transpose(2, 0, 1)[None].copy()


class DatasetReader(CalibrationDataReader):
    def __init__(self, input_name: str, paths: list[Path]):
        self._it = iter([{input_name: preprocess(cv2.imread(str(p)))} for p in paths])

    def get_next(self):
        return next(self._it, None)


def main():
    src = fr.find_model("w600k_mbf.onnx")
    dst = ROOT / "uno_q" / "models" / "w600k_mbf_int8.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "dataset").glob("*/*.jpg"))
    if not paths:
        sys.exit("No calibration images in dataset/. Run scripts/collect_dataset.py first.")

    with tempfile.TemporaryDirectory() as tmp:
        op13 = Path(tmp) / "mbf_op13.onnx"
        pre = Path(tmp) / "mbf_pre.onnx"
        onnx.save(version_converter.convert_version(onnx.load(src), 13), op13)
        quant_pre_process(str(op13), str(pre))
        model = onnx.load(pre)
        head = [n.name for n in model.graph.node if n.op_type in ("Flatten", "Gemm", "BatchNormalization")]
        input_name = model.graph.input[0].name
        quantize_static(str(pre), str(dst), DatasetReader(input_name, paths[::2]),
                        quant_format=QuantFormat.QOperator, per_channel=False,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                        nodes_to_exclude=head)

    fp32 = fr.MobileFaceNet(src, 4)
    int8 = fr.MobileFaceNet(str(dst), 4)
    cos = [float(fp32.embed(img) @ int8.embed(img)) for img in (cv2.imread(str(p)) for p in paths[::5])]
    print(f"Saved {dst.relative_to(ROOT)} ({dst.stat().st_size // 1024} KB, FP32 is "
          f"{Path(src).stat().st_size // 1024} KB)")
    print(f"Mean cosine(FP32, INT8) on dataset: {np.mean(cos):.3f}  -> enroll and recognise with the same model")
    print(f"ONNX Runtime {ort.__version__}. Next: evaluate.py --model and benchmark_edge.py --int8 on the UNO Q.")


if __name__ == "__main__":
    main()
