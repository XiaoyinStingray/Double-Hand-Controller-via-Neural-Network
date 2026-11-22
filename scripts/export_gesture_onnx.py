"""Export a gesture classifier to ONNX and benchmark it with onnxruntime.

Usage:
    python scripts/export_gesture_onnx.py --checkpoint runs/gesture_transformer/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.gesture import build_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--runs", type=int, default=200)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt["model"]
    classes = ckpt["classes"]
    T = int(ckpt["meta"]["T"])
    in_dim = int(ckpt["meta"]["feature_dim"])
    model = build_model(model_name, in_dim=in_dim, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # PyTorch's fused transformer encoder fast path cannot be exported to ONNX.
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        pass

    out = args.out or (ROOT / "runs" / "export" / f"gesture_{model_name}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, T, in_dim)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(out),
            input_names=["features"], output_names=["logits"],
            opset_version=args.opset,
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(out))

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.randn(8, T, in_dim).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"features": x})[0]
    max_diff = float(np.abs(ref - got).max())

    x1 = np.random.randn(1, T, in_dim).astype(np.float32)
    for _ in range(10):
        sess.run(None, {"features": x1})
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, {"features": x1})
        times.append((time.perf_counter() - t0) * 1000)
    times = np.asarray(times)
    p50, p95 = float(np.percentile(times, 50)), float(np.percentile(times, 95))

    meta = {
        "model": model_name,
        "opset": args.opset,
        "T": T,
        "feature_dim": in_dim,
        "classes": classes,
        "max_abs_diff": max_diff,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "runs": args.runs,
        "size_mb": out.stat().st_size / 1e6,
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"model {model_name} | T={T} | in_dim={in_dim} | classes={len(classes)}")
    print(f"onnx {out} ({meta['size_mb']:.2f} MB) | max_abs_diff {max_diff:.2e}")
    print(f"cpu latency p50 {p50:.3f} ms | p95 {p95:.3f} ms ({args.runs} runs, batch=1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
