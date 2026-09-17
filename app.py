#!/usr/bin/env python3
"""智能检测算法验证台 — 原始模型剪枝+INT8 · 农业病虫害监测（演示 API，无真实权重推理）"""

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
ORIGINAL_BASE = {
    "conf": 0.95,
    "params": 25.8,
    "sizeMB": 6.6,
    "inferMs": 48.0,
    "fps": 20.8,
    "precision": 97.6,
    "recall": 90.7,
    "speedGain": 0.0,
}
YOLO_BASE = ORIGINAL_BASE  # backward-compat alias

# Shared geometry: same count/position/label; only conf differs
SAMPLE_GEOM = {
    "normal": [
        {"id": "n1", "label": "Leaf Spot", "x": 42, "y": 28, "w": 40, "h": 42, "yoloConf": 0.95, "prunedConf": 0.85},
        {"id": "n2", "label": "Gray Mold", "x": 18, "y": 18, "w": 24, "h": 28, "yoloConf": 0.91, "prunedConf": 0.82},
    ],
    "small": [
        {"id": "s1", "label": "Anthracnose Fruit Rot", "x": 44, "y": 40, "w": 16, "h": 18, "yoloConf": 0.88, "prunedConf": 0.80},
        {"id": "s2", "label": "Blossom Blight", "x": 58, "y": 48, "w": 12, "h": 12, "yoloConf": 0.84, "prunedConf": 0.76},
    ],
    "occlusion": [
        {"id": "o1", "label": "Powdery Mildew Leaf", "x": 36, "y": 28, "w": 28, "h": 36, "yoloConf": 0.90, "prunedConf": 0.82},
        {"id": "o2", "label": "Powdery Mildew Fruit", "x": 12, "y": 40, "w": 22, "h": 24, "yoloConf": 0.86, "prunedConf": 0.79},
    ],
    "lowlight": [
        {"id": "l1", "label": "Powdery Mildew Fruit", "x": 30, "y": 28, "w": 36, "h": 34, "yoloConf": 0.87, "prunedConf": 0.81},
        {"id": "l2", "label": "Leaf Spot", "x": 58, "y": 52, "w": 20, "h": 18, "yoloConf": 0.83, "prunedConf": 0.76},
    ],
    "complex": [
        {"id": "c1", "label": "Anthracnose Fruit Rot", "x": 36, "y": 48, "w": 28, "h": 26, "yoloConf": 0.92, "prunedConf": 0.84},
        {"id": "c2", "label": "Blossom Blight", "x": 48, "y": 28, "w": 26, "h": 24, "yoloConf": 0.89, "prunedConf": 0.80},
    ],
    "req_a": [
        {"id": "ra1", "label": "Anthracnose Fruit Rot", "x": 28, "y": 24, "w": 44, "h": 48, "yoloConf": 0.95, "prunedConf": 0.85},
        {"id": "ra2", "label": "Fruit Lesion", "x": 52, "y": 58, "w": 22, "h": 20, "yoloConf": 0.90, "prunedConf": 0.80},
    ],
    "req_b": [
        {"id": "rb1", "label": "Fruit Lesion", "x": 30, "y": 26, "w": 42, "h": 46, "yoloConf": 0.94, "prunedConf": 0.84},
        {"id": "rb2", "label": "Anthracnose Fruit Rot", "x": 18, "y": 54, "w": 20, "h": 18, "yoloConf": 0.88, "prunedConf": 0.79},
    ],
}


def _boxes_from_geom(geom: list, kind: str) -> list:
    key = "yoloConf" if kind == "yolo" else "prunedConf"
    return [
        {
            "id": b["id"],
            "label": b["label"],
            "x": b["x"],
            "y": b["y"],
            "w": b["w"],
            "h": b["h"],
            "baseConf": b[key],
        }
        for b in geom
    ]


SAMPLE_BOXES = {
    sid: {"yolo": _boxes_from_geom(g, "yolo"), "pruned": _boxes_from_geom(g, "pruned")}
    for sid, g in SAMPLE_GEOM.items()
}

SYNTH_LABELS = ["Anthracnose Fruit Rot", "Fruit Lesion", "Leaf Spot"]


def synthesize_boxes(kind: str) -> list:
    """Same geometry for both models; only conf differs."""
    geom = [
        {"id": "u1", "label": "Anthracnose Fruit Rot", "x": 30, "y": 26, "w": 40, "h": 44, "yoloConf": 0.95, "prunedConf": 0.85},
        {"id": "u2", "label": "Fruit Lesion", "x": 52, "y": 54, "w": 20, "h": 18, "yoloConf": 0.91, "prunedConf": 0.81},
    ]
    return _boxes_from_geom(geom, kind)


def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def pruned_metrics(prune_ratio: float) -> dict:
    """Mock metrics: conf 0.85, size 3.3MB (−50%), speed +40%, precision ~87.8 at 0.2."""
    r = clamp(prune_ratio, 0.05, 0.5)
    d = r - 0.2
    return {
        "conf": round(clamp(0.85 - d * 0.4, 0.72, 0.93), 2),
        "params": round(clamp(12.9 - d * 18, 8.0, 18.0), 2),
        "sizeMB": round(clamp(3.3 - d * 5, 2.2, 5.0), 1),
        "inferMs": round(clamp(34.3 + d * 18, 26.0, 45.0), 1),
        "fps": round(clamp(29.2 - d * 12, 20.0, 36.0), 1),
        "precision": round(clamp(87.8 - d * 12, 80.0, 94.0), 1),
        "recall": round(clamp(86.0 - d * 10, 78.0, 92.0), 1),
        "speedGain": round(clamp(40 - d * 40, 20, 55), 0),
    }


def conf_display(base: float, threshold: float) -> float:
    delta = (threshold - 0.52) * 0.15
    return round(clamp(base + delta, 0.45, 0.99), 3)


def sample_from_path(path: str) -> str:
    m = re.search(r"(normal|small|occlusion|lowlight|complex|req_a|req_b)", path or "")
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

    def _send_bytes(self, data: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._send_bytes(b'{"ok":true}', "application/json; charset=utf-8")
            return
        if path == "/api/infer":
            self._handle_infer(parsed)
            return
        # Explicit file serve — avoids SimpleHTTP unicode-directory 404s
        if path in ("/", "/index.html"):
            fp = ROOT / "index.html"
            if not fp.is_file():
                self.send_error(404, "index.html missing")
                return
            self._send_bytes(fp.read_bytes(), "text/html; charset=utf-8")
            return
        # Other static assets under ROOT
        rel = path.lstrip("/")
        if not rel or ".." in rel.split("/"):
            self.send_error(404)
            return
        fp = (ROOT / rel).resolve()
        try:
            fp.relative_to(ROOT)
        except ValueError:
            self.send_error(403)
            return
        if not fp.is_file():
            self.send_error(404)
            return
        ctype = self.guess_type(str(fp))
        self._send_bytes(fp.read_bytes(), ctype)

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
        known = sample in SAMPLE_BOXES
        if not known:
            # uploaded / unknown image → synthesize boxes; keep sample id for client
            sample = sample or "upload"
        prune_ratio = clamp(float(body.get("prune_ratio", body.get("compress", 0.2))), 0.05, 0.5)
        threshold = clamp(float(body.get("threshold", 0.52)), 0.3, 0.7)
        mode = body.get("mode", "single")
        script = body.get("script", "prune")
        metrics = pruned_metrics(prune_ratio)
        original_name = body.get("original_model") or "bch.pt"
        pruned_name = body.get("pruned_model") or "bch_int8.onnx"
        image_name = body.get("image_name") or (f"samples/{sample}.jpg" if known else "upload.jpg")

        def annotate(boxes):
            out = []
            for b in boxes:
                item = dict(b)
                item["conf"] = conf_display(b["baseConf"], threshold)
                out.append(item)
            return out

        if known:
            boxes = SAMPLE_BOXES[sample]
            yolo_boxes = boxes["yolo"]
            pruned_boxes = boxes["pruned"]
        else:
            yolo_boxes = synthesize_boxes("yolo")
            pruned_boxes = synthesize_boxes("pruned")

        if script == "validate":
            logs = [
                "$ python validate_testset.py",
                f"加载 data.yaml / weights/{original_name}…",
                f"模型加载成功: ./weights/{original_name}",
                "开始测试集验证 (split=test)…",
                f"done. conf={metrics['conf']:.2f}  size={metrics['sizeMB']}MB  speed=+{int(metrics['speedGain'])}%",
            ]
        else:
            logs = [
                "$ python prune_quantize.py",
                f"Loading 原始模型('./weights/{original_name}')…",
                f"L1 unstructured prune Conv2d (prune_ratio={prune_ratio:.2f})…",
                "Export ONNX (opset=17, simplify=True)…",
                f"quantize_dynamic → ./weights/{pruned_name}",
                "剪枝+INT8量化完成",
                f"mock infer {image_name}  conf={metrics['conf']:.2f}  size={metrics['sizeMB']}MB  speed=+{int(metrics['speedGain'])}%",
            ]

        # Dual-model validate timing hints for the client wizard
        timing = {
            "original_ms": 2000,
            "pruned_ms": 1200,
        }

        payload = {
            "ok": True,
            "sample": sample if known else "upload",
            "prune_ratio": prune_ratio,
            "threshold": threshold,
            "mode": mode,
            "script": script,
            "logs": logs,
            "timing": timing,
            "charts": {
                "baseline": "charts/curve_baseline.png",
                "pruned": "charts/curve_pruned.png",
            },
            "excel": {
                "conf_original": 0.95,
                "conf_yolo": 0.95,
                "conf_pruned": 0.85,
                "speed_gain_pct": 40,
                "size_original_mb": 6.6,
                "size_yolo_mb": 6.6,
                "size_pruned_mb": 3.3,
                "size_drop_pct": 50,
                "precision_original": 97.6,
                "precision_pruned": 87.8,
                "classes": ["Angular Leafspot", "Anthracnose Fruit Rot", "Blossom Blight", "Gray Mold", "Leaf Spot", "Powdery Mildew Fruit", "Powdery Mildew Leaf", "Anthracnose Fruit Rot", "Fruit Lesion"],
            },
            "pruned": {
                "metrics": metrics,
                "boxes": annotate(pruned_boxes),
            },
            "qrg": {
                "metrics": metrics,
                "boxes": annotate(pruned_boxes),
            },
            "yolo": {
                "metrics": ORIGINAL_BASE,
                "boxes": annotate(yolo_boxes),
            },
            "original": {
                "metrics": ORIGINAL_BASE,
                "boxes": annotate(yolo_boxes),
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
