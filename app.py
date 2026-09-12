#!/usr/bin/env python3
"""智能检测算法验证台 — YOLO剪枝+INT8 · 农业病虫害监测（演示 API，无真实权重推理）"""

from __future__ import annotations

import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8765"))

# Excel-aligned bold metrics (草莓病害)
YOLO_BASE = {
    "conf": 0.95,
    "params": 25.8,
    "sizeMB": 6.6,
    "inferMs": 48.0,
    "fps": 20.8,
    "precision": 97.6,
    "recall": 90.7,
    "speedGain": 0.0,
}

SAMPLE_BOXES = {
    "normal": {
        "yolo": [
            {"id": "n1", "label": "叶斑病", "x": 38, "y": 22, "w": 34, "h": 48, "baseConf": 0.95},
            {"id": "n2", "label": "灰霉病", "x": 14, "y": 16, "w": 20, "h": 26, "baseConf": 0.91},
        ],
        "pruned": [
            {"id": "n1", "label": "叶斑病", "x": 36, "y": 20, "w": 36, "h": 50, "baseConf": 0.85},
            {"id": "n2", "label": "灰霉病", "x": 12, "y": 14, "w": 22, "h": 28, "baseConf": 0.82},
            {"id": "n3", "label": "角斑病", "x": 52, "y": 58, "w": 16, "h": 14, "baseConf": 0.78},
        ],
    },
    "small": {
        "yolo": [{"id": "s1", "label": "炭疽病", "x": 32, "y": 38, "w": 14, "h": 16, "baseConf": 0.88}],
        "pruned": [
            {"id": "s1", "label": "炭疽病", "x": 30, "y": 36, "w": 16, "h": 18, "baseConf": 0.80},
            {"id": "s2", "label": "花枯病", "x": 48, "y": 42, "w": 10, "h": 12, "baseConf": 0.76},
            {"id": "s3", "label": "叶斑病", "x": 22, "y": 52, "w": 8, "h": 9, "baseConf": 0.74},
        ],
    },
    "occlusion": {
        "yolo": [{"id": "o1", "label": "白粉病叶片", "x": 35, "y": 30, "w": 22, "h": 32, "baseConf": 0.90}],
        "pruned": [
            {"id": "o1", "label": "白粉病叶片", "x": 10, "y": 34, "w": 24, "h": 26, "baseConf": 0.82},
            {"id": "o2", "label": "白粉病果实", "x": 36, "y": 28, "w": 22, "h": 34, "baseConf": 0.79},
            {"id": "o3", "label": "灰霉病", "x": 60, "y": 38, "w": 24, "h": 24, "baseConf": 0.77},
        ],
    },
    "lowlight": {
        "yolo": [{"id": "l1", "label": "白粉病果实", "x": 32, "y": 36, "w": 28, "h": 26, "baseConf": 0.87}],
        "pruned": [
            {"id": "l1", "label": "白粉病果实", "x": 28, "y": 32, "w": 34, "h": 30, "baseConf": 0.81},
            {"id": "l2", "label": "叶斑病", "x": 58, "y": 48, "w": 18, "h": 20, "baseConf": 0.76},
            {"id": "l3", "label": "角斑病", "x": 40, "y": 62, "w": 16, "h": 14, "baseConf": 0.74},
        ],
    },
    "complex": {
        "yolo": [
            {"id": "c1", "label": "炭疽病", "x": 40, "y": 68, "w": 22, "h": 20, "baseConf": 0.92},
            {"id": "c2", "label": "花枯病", "x": 48, "y": 52, "w": 24, "h": 22, "baseConf": 0.89},
        ],
        "pruned": [
            {"id": "c1", "label": "炭疽病", "x": 38, "y": 66, "w": 24, "h": 22, "baseConf": 0.84},
            {"id": "c2", "label": "花枯病", "x": 46, "y": 48, "w": 26, "h": 24, "baseConf": 0.80},
            {"id": "c3", "label": "叶斑病", "x": 22, "y": 38, "w": 20, "h": 26, "baseConf": 0.77},
        ],
    },
}


def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def pruned_metrics(prune_ratio: float) -> dict:
    """Mock metrics aligned to Excel: conf 0.85, size 3.6MB, speed +15% at 0.2."""
    r = clamp(prune_ratio, 0.05, 0.5)
    d = r - 0.2
    return {
        "conf": round(clamp(0.85 - d * 0.4, 0.72, 0.93), 2),
        "params": round(clamp(14.2 - d * 20, 9.0, 20.0), 2),
        "sizeMB": round(clamp(3.6 - d * 6, 2.4, 5.5), 1),
        "inferMs": round(clamp(41.7 + d * 20, 32.0, 48.0), 1),
        "fps": round(clamp(24.0 - d * 12, 18.0, 30.0), 1),
        "precision": round(clamp(92.0 - d * 12, 85.0, 96.0), 1),
        "recall": round(clamp(88.6 - d * 10, 82.0, 93.0), 1),
        "speedGain": round(clamp(15 - d * 40, 5, 28), 0),
    }


def conf_display(base: float, threshold: float) -> float:
    delta = (threshold - 0.52) * 0.15
    return round(clamp(base + delta, 0.45, 0.99), 3)


def sample_from_path(path: str) -> str:
    m = re.search(r"(normal|small|occlusion|lowlight|complex)", path or "")
    return m.group(1) if m else "normal"


class Handler(SimpleHTTPRequestHandler):
    # Render/proxy health checks need HTTP/1.1 (stdlib default is 1.0)
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        # Keep ASCII-only — Render free runners often use a C locale
        try:
            print(f"[bench] {args[0]}", flush=True)
        except Exception:
            pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            data = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/infer":
            self._handle_infer(parsed)
            return
        if parsed.path in ("/", "/index.html"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/infer":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                body = {}
            self._respond_infer(body)
            return
        self.send_error(404)

    def _handle_infer(self, parsed) -> None:
        qs = parse_qs(parsed.query)
        body = {
            "sample": (qs.get("sample") or ["normal"])[0],
            "prune_ratio": float((qs.get("prune_ratio") or ["0.2"])[0]),
            "threshold": float((qs.get("threshold") or ["0.52"])[0]),
            "mode": (qs.get("mode") or ["single"])[0],
            "script": (qs.get("script") or ["prune"])[0],
        }
        self._respond_infer(body)

    def _respond_infer(self, body: dict) -> None:
        sample = body.get("sample") or sample_from_path(str(body.get("path", "")))
        if sample not in SAMPLE_BOXES:
            sample = "normal"
        prune_ratio = clamp(float(body.get("prune_ratio", body.get("compress", 0.2))), 0.05, 0.5)
        threshold = clamp(float(body.get("threshold", 0.52)), 0.3, 0.7)
        mode = body.get("mode", "single")
        script = body.get("script", "prune")
        metrics = pruned_metrics(prune_ratio)

        def annotate(boxes):
            out = []
            for b in boxes:
                item = dict(b)
                item["conf"] = conf_display(b["baseConf"], threshold)
                out.append(item)
            return out

        boxes = SAMPLE_BOXES[sample]
        if script == "validate":
            logs = [
                "$ python validate_testset.py",
                "加载 data.yaml / weights/bch.pt…",
                "模型加载成功: ./weights/bch.pt",
                "开始测试集验证 (split=test)…",
                f"done. conf={metrics['conf']:.2f}  size={metrics['sizeMB']}MB  speed=+{int(metrics['speedGain'])}%",
            ]
        else:
            logs = [
                "$ python prune_quantize.py",
                "Loading YOLO('./weights/bch.pt')…",
                f"L1 unstructured prune Conv2d (prune_ratio={prune_ratio:.2f})…",
                "Export ONNX (opset=17, simplify=True)…",
                "quantize_dynamic → ./weights/bch_int8.onnx",
                "剪枝+INT8量化完成",
                f"mock infer samples/{sample}.jpg  conf={metrics['conf']:.2f}  size={metrics['sizeMB']}MB  speed=+{int(metrics['speedGain'])}%",
            ]

        payload = {
            "ok": True,
            "sample": sample,
            "prune_ratio": prune_ratio,
            "threshold": threshold,
            "mode": mode,
            "script": script,
            "logs": logs,
            "charts": {
                "baseline": "charts/curve_baseline.png",
                "pruned": "charts/curve_pruned.png",
            },
            "excel": {
                "conf_yolo": 0.95,
                "conf_pruned": 0.85,
                "speed_gain_pct": 15,
                "size_yolo_mb": 6.6,
                "size_pruned_mb": 3.6,
                "classes": ["角斑病", "炭疽病", "花枯病", "灰霉病", "叶斑病", "白粉病果实", "白粉病叶片"],
            },
            "pruned": {
                "metrics": metrics,
                "boxes": annotate(boxes["pruned"]),
            },
            "qrg": {
                "metrics": metrics,
                "boxes": annotate(boxes["pruned"]),
            },
            "yolo": {
                "metrics": YOLO_BASE,
                "boxes": annotate(boxes["yolo"]),
            },
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("algo-validation-bench ready", flush=True)
    print(f"  listening on 0.0.0.0:{PORT}", flush=True)
    print("  API  POST/GET /api/infer  GET /api/health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()


if __name__ == "__main__":
    main()
