#!/usr/bin/env python3
"""
Fingerprint & Image Classification App — Kivy Edition
======================================================
A user-friendly mobile/desktop app for:
  • Fingerprint classification (Arch, Tented Arch, Left/Right Loop, Whorl variants)
  • General image classification (auto-detect fingerprint vs photo vs document etc.)
  • Camera capture for phone fingerprint scans
  • Scan history with export
  • Responsive layout for desktop & mobile

Dependencies: kivy, opencv-python, numpy
Optional (Android): plyer (for camera)
"""

import time, os, json, threading, math, hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.slider import Slider
from kivy.uix.image import Image as KivyImage
from kivy.uix.widget import Widget
from kivy.uix.scatter import Scatter
from kivy.uix.camera import Camera
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Line, RoundedRectangle, Ellipse
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.properties import (
    StringProperty, NumericProperty, ListProperty,
    ObjectProperty, BooleanProperty, DictProperty, AliasProperty,
)
from kivy.lang import Builder
from kivy.config import Config
from kivy.core.window import Window
from kivy.animation import Animation

# ── Window Configuration ────────────────────────────────────────────────
Config.set("graphics", "width", "420")
Config.set("graphics", "height", "780")
Config.set("graphics", "resizable", "1")
Config.set("input", "mouse", "mouse,multitouch_on_demand")

HISTORY_DIR = Path.home() / ".fingerprint_app"
HISTORY_FILE = HISTORY_DIR / "history.json"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════

def cv2_to_texture(cv_img):
    if cv_img is None:
        return None
    img = np.ascontiguousarray(cv_img)
    if len(img.shape) == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return None
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    buf = rgb.tobytes()
    texture = Texture.create(size=(w, h), colorfmt="rgb")
    texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")
    texture.flip_vertical()
    return texture


def save_cv_image(cv_img, path):
    """Save an OpenCV image to disk."""
    if cv_img is None:
        return False
    cv2.imwrite(str(path), cv_img)
    return True


# ═══════════════════════════════════════════════════════════════════════
# HISTORY MANAGER
# ═══════════════════════════════════════════════════════════════════════

class HistoryManager:
    """Simple JSON-based scan history."""

    def __init__(self):
        self.records = []
        self._load()

    def _load(self):
        if HISTORY_FILE.exists():
            try:
                self.records = json.loads(HISTORY_FILE.read_text())
            except Exception:
                self.records = []

    def _save(self):
        HISTORY_FILE.write_text(json.dumps(self.records[-200:], indent=2))

    def add(self, classification, confidence, quality, source, thumb_path=None):
        rec = {
            "id": hashlib.md5(f"{time.time()}{classification}".encode()).hexdigest()[:8],
            "timestamp": datetime.now().isoformat(),
            "classification": classification,
            "confidence": round(confidence, 3),
            "quality": round(quality, 3),
            "source": source,
            "thumb_path": str(thumb_path) if thumb_path else None,
        }
        self.records.append(rec)
        self._save()
        return rec

    def clear(self):
        self.records = []
        self._save()

    def recent(self, n=20):
        return self.records[-n:][::-1]


history_mgr = HistoryManager()


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC FINGERPRINT GENERATOR
# ═══════════════════════════════════════════════════════════════════════

class FingerprintGenerator:
    PATTERNS = ["Arch", "Tented Arch", "Left Loop", "Right Loop", "Whorl"]

    @staticmethod
    def generate(size=300, pattern="Whorl", noise_level=0.15, seed=None):
        if seed is not None:
            np.random.seed(seed)
        dispatch = {
            "Arch": FingerprintGenerator.generate_arch,
            "Tented Arch": FingerprintGenerator.generate_tented_arch,
            "Left Loop": lambda s, n: FingerprintGenerator.generate_loop(s, "left", n),
            "Right Loop": lambda s, n: FingerprintGenerator.generate_loop(s, "right", n),
            "Whorl": FingerprintGenerator.generate_whorl,
        }
        fn = dispatch.get(pattern, FingerprintGenerator.generate_whorl)
        return fn(size, noise_level)

    @staticmethod
    def generate_arch(size=300, noise_level=0.15):
        x = np.linspace(0, 4 * np.pi, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        center_rise = 0.8 * np.exp(-Y ** 2 / 2.0)
        pattern = np.sin(X * 3 + center_rise * np.sin(X * 0.5))
        pattern += np.random.randn(size, size) * noise_level
        pattern = cv2.normalize(pattern, None, 0, 255, cv2.NORM_MINMAX)
        pattern = np.clip(pattern, 0, 255).astype(np.uint8)
        mask = FingerprintGenerator._fingerprint_mask(size)
        pattern = cv2.bitwise_and(pattern, mask)
        return cv2.GaussianBlur(pattern, (3, 3), 0.5)

    @staticmethod
    def generate_tented_arch(size=300, noise_level=0.15):
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x, y)
        upthrust = 3.0 * np.exp(-(X ** 2) / 0.5)
        pattern = np.sin(Y * 5 + upthrust)
        pattern += np.random.randn(size, size) * noise_level
        pattern = cv2.normalize(pattern, None, 0, 255, cv2.NORM_MINMAX)
        pattern = np.clip(pattern, 0, 255).astype(np.uint8)
        mask = FingerprintGenerator._fingerprint_mask(size)
        pattern = cv2.bitwise_and(pattern, mask)
        return cv2.GaussianBlur(pattern, (3, 3), 0.5)

    @staticmethod
    def generate_loop(size=300, direction="left", noise_level=0.15):
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x, y)
        cx = -0.5 if direction == "left" else 0.5
        cy = -0.5
        theta = np.arctan2(Y - cy, X - cx)
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        loop_pattern = np.sin(theta * 3 + r * 5)
        parallel_pattern = np.sin(Y * 5)
        blend = np.clip(r / 2.0, 0, 1)
        pattern = loop_pattern * (1 - blend) + parallel_pattern * blend
        pattern += np.random.randn(size, size) * noise_level
        pattern = cv2.normalize(pattern, None, 0, 255, cv2.NORM_MINMAX)
        pattern = np.clip(pattern, 0, 255).astype(np.uint8)
        mask = FingerprintGenerator._fingerprint_mask(size)
        pattern = cv2.bitwise_and(pattern, mask)
        return cv2.GaussianBlur(pattern, (3, 3), 0.5)

    @staticmethod
    def generate_whorl(size=300, noise_level=0.15):
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)
        pattern = np.sin(r * 6 + theta * 0.5)
        pattern += np.random.randn(size, size) * noise_level
        pattern = cv2.normalize(pattern, None, 0, 255, cv2.NORM_MINMAX)
        pattern = np.clip(pattern, 0, 255).astype(np.uint8)
        mask = FingerprintGenerator._fingerprint_mask(size)
        pattern = cv2.bitwise_and(pattern, mask)
        return cv2.GaussianBlur(pattern, (3, 3), 0.5)

    @staticmethod
    def _fingerprint_mask(size):
        mask = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        axes = (int(size * 0.42), int(size * 0.46))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        mask[mask > 128] = 255
        mask[mask <= 128] = 0
        return mask


# ═══════════════════════════════════════════════════════════════════════
# FINGERPRINT PROCESSING ENGINE
# ═══════════════════════════════════════════════════════════════════════

class FingerprintProcessor:
    """Core fingerprint image processing and classification engine."""

    ARCH = "Arch"
    TENTED_ARCH = "Tented Arch"
    LEFT_LOOP = "Left Loop"
    RIGHT_LOOP = "Right Loop"
    PLAIN_WHORL = "Plain Whorl"
    DOUBLE_LOOP_WHORL = "Double Loop Whorl"
    CENTRAL_POCKET = "Central Pocket Whorl"
    UNKNOWN = "Unknown"

    STEPS = [
        "Preprocessing", "Segmentation", "Orientation Field",
        "Ridge Enhancement", "Binarization", "Thinning",
        "Singular Point Detection", "Classification", "Quality Assessment",
    ]

    def __init__(self):
        self.block_size = 16
        self.original = None
        self.gray = None
        self.enhanced = None
        self.clahe_enhanced = None
        self.orientation_map = None
        self.orientation_strength = None
        self.binary_img = None
        self.thinned = None
        self.segmentation_mask = None
        self.cores = []
        self.deltas = []
        self.classification = self.UNKNOWN
        self.confidence = 0.0
        self.details = {}
        self.quality_score = 0.0
        self.processing_time = 0.0
        self.image_info = {}
        self.file_path = None
        self.current_step = ""
        self.step_callback = None

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        max_dim = 800
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.original = img.copy()
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        self.file_path = path
        self._update_image_info()
        self._reset_results()
        return True

    def load_from_array(self, gray_img, label="camera"):
        self.gray = gray_img.copy()
        self.original = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR) if len(gray_img.shape) == 2 else gray_img.copy()
        self.file_path = label
        self._update_image_info()
        self._reset_results()
        return True

    def _update_image_info(self):
        if self.gray is not None:
            h, w = self.gray.shape
            self.image_info = {
                "width": w, "height": h, "size_str": f"{w} × {h}",
                "pixels": w * h,
                "mean_intensity": float(np.mean(self.gray)),
                "std_intensity": float(np.std(self.gray)),
                "source": self.file_path or "unknown",
            }
        else:
            self.image_info = {}

    def _reset_results(self):
        self.enhanced = self.clahe_enhanced = self.orientation_map = None
        self.orientation_strength = self.binary_img = self.thinned = None
        self.segmentation_mask = None
        self.cores, self.deltas = [], []
        self.classification, self.confidence = self.UNKNOWN, 0.0
        self.details, self.quality_score, self.processing_time = {}, 0.0, 0.0
        self.current_step = ""

    def process(self, block_size=16):
        if self.gray is None:
            raise ValueError("No image loaded")
        self.block_size = max(8, block_size)
        self._reset_results()
        t0 = time.perf_counter()
        steps = [
            self._preprocess, self._segment, self._compute_orientation,
            self._enhance_ridges, self._binarize, self._thin,
            self._detect_singular_points, self._classify, self._assess_quality,
        ]
        for i, fn in enumerate(steps):
            self.current_step = self.STEPS[i]
            if self.step_callback:
                self.step_callback(self.current_step, i)
            fn()
        self.processing_time = time.perf_counter() - t0
        self.details["processing_time"] = self.processing_time

    # ── Individual processing steps ──────────────────────────────────
    def _preprocess(self):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.clahe_enhanced = clahe.apply(self.gray)
        self.clahe_enhanced = cv2.normalize(self.clahe_enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self.enhanced = self.clahe_enhanced.copy()

    def _segment(self):
        h, w = self.gray.shape
        bh, bw = self.block_size, self.block_size
        mask = np.zeros_like(self.gray, dtype=np.float32)
        for y in range(0, h - bh + 1, bh):
            for x in range(0, w - bw + 1, bw):
                block = self.gray[y:y + bh, x:x + bw]
                var, mean = np.var(block), np.mean(block)
                mask[y:y + bh, x:x + bw] = min(var / 500.0, 1.0) if 30 < mean < 220 else 0.0
        mask = cv2.GaussianBlur(mask, (self.block_size * 2 + 1,) * 2, 0)
        self.segmentation_mask = (mask > 0.2).astype(np.uint8) * 255
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh, bw))
        self.segmentation_mask = cv2.morphologyEx(self.segmentation_mask, cv2.MORPH_CLOSE, kern, iterations=1)
        self.segmentation_mask = cv2.morphologyEx(self.segmentation_mask, cv2.MORPH_OPEN, kern, iterations=1)

    def _compute_orientation(self):
        h, w = self.enhanced.shape
        bh, bw = self.block_size, self.block_size
        gx = cv2.GaussianBlur(cv2.Sobel(self.enhanced, cv2.CV_32F, 1, 0, ksize=3), (5, 5), 0)
        gy = cv2.GaussianBlur(cv2.Sobel(self.enhanced, cv2.CV_32F, 0, 1, ksize=3), (5, 5), 0)
        rows, cols = h // bh, w // bw
        self.orientation_map = np.zeros((rows, cols), dtype=np.float32)
        self.orientation_strength = np.zeros((rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                y1, y2, x1, x2 = i * bh, (i + 1) * bh, j * bw, (j + 1) * bw
                bgx, bgy = gx[y1:y2, x1:x2], gy[y1:y2, x1:x2]
                gxx, gyy, gxy = np.sum(bgx ** 2), np.sum(bgy ** 2), np.sum(bgx * bgy)
                denom = gxx + gyy
                if denom < 1e-10:
                    continue
                self.orientation_map[i, j] = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
                self.orientation_strength[i, j] = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / denom
        self._smooth_orientation()

    def _smooth_orientation(self):
        h, w = self.orientation_map.shape
        sin2, cos2 = np.sin(2 * self.orientation_map), np.cos(2 * self.orientation_map)
        ksz, pad = 5, 2
        sin2_pad, cos2_pad = np.pad(sin2, pad, mode="edge"), np.pad(cos2, pad, mode="edge")
        smoothed = np.zeros_like(self.orientation_map)
        for i in range(h):
            for j in range(w):
                ip, jp = i + pad, j + pad
                avg_sin = np.mean(sin2_pad[ip - pad:ip + pad + 1, jp - pad:jp + pad + 1])
                avg_cos = np.mean(cos2_pad[ip - pad:ip + pad + 1, jp - pad:jp + pad + 1])
                smoothed[i, j] = 0.5 * np.arctan2(avg_sin, avg_cos)
        self.orientation_map = smoothed

    def _enhance_ridges(self):
        if self.orientation_map is None:
            return
        freq = self._estimate_ridge_frequency()
        ksize = max(21, self.block_size * 2 + 1)
        n_ori, orientations = 12, np.arange(12) * np.pi / 12
        filtered = [
            np.abs(cv2.filter2D(self.clahe_enhanced, cv2.CV_32F,
                                cv2.getGaborKernel((ksize, ksize), 4.0, t, freq, 0.5, 0, ktype=cv2.CV_32F)))
            for t in orientations
        ]
        h, w = self.clahe_enhanced.shape
        rows, cols = self.orientation_map.shape
        result = np.zeros((h, w), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                if self.orientation_strength[i, j] < 0.05:
                    continue
                y1, y2 = i * self.block_size, min((i + 1) * self.block_size, h)
                x1, x2 = j * self.block_size, min((j + 1) * self.block_size, w)
                theta = self.orientation_map[i, j] % np.pi
                if theta < 0:
                    theta += np.pi
                idx = int(round(theta / np.pi * n_ori)) % n_ori
                candidates = [(idx - 1) % n_ori, idx, (idx + 1) % n_ori]
                best = max(candidates, key=lambda k: float(np.mean(filtered[k][y1:y2, x1:x2])))
                result[y1:y2, x1:x2] = filtered[best][y1:y2, x1:x2]
        if self.segmentation_mask is not None:
            result *= self.segmentation_mask.astype(np.float32) / 255.0
        self.enhanced = np.clip(cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX), 0, 255).astype(np.uint8)

    def _estimate_ridge_frequency(self):
        h, w = self.enhanced.shape
        rs = min(h, w) // 3
        if rs < 4:
            return 0.1
        region = self.enhanced[h // 2 - rs // 2:h // 2 + rs // 2, w // 2 - rs // 2:w // 2 + rs // 2]
        if region.size == 0:
            return 0.1
        estimates = []
        for sig in [np.mean(region, axis=0), np.mean(region, axis=1)]:
            n = len(sig)
            if n < 4:
                continue
            ac = np.correlate(sig - np.mean(sig), sig - np.mean(sig), mode="full")[n - 1:]
            for k in range(3, len(ac) // 2):
                if ac[k] > ac[k - 1] and ac[k] > ac[k + 1] and ac[k] > 0.1 * ac[0]:
                    estimates.append(1.0 / max(k, 3))
                    break
        return float(np.median(estimates)) if estimates else 0.1

    def _binarize(self):
        self.binary_img = cv2.adaptiveThreshold(
            self.enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )
        if self.segmentation_mask is not None:
            self.binary_img = cv2.bitwise_and(self.binary_img, self.segmentation_mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.binary_img = cv2.morphologyEx(cv2.morphologyEx(self.binary_img, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)

    def _thin(self):
        if self.binary_img is None:
            return
        binary = (self.binary_img // 255).astype(np.uint8)
        skel, img = np.zeros_like(binary), binary.copy()
        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for _ in range(200):
            if np.count_nonzero(img) == 0:
                break
            eroded = cv2.erode(img, k)
            skel = cv2.bitwise_or(skel, cv2.subtract(img, cv2.dilate(eroded, k)))
            img = eroded
        self.thinned = skel * 255

    def _detect_singular_points(self):
        if self.orientation_map is None:
            return
        rows, cols = self.orientation_map.shape
        self.cores, self.deltas = [], []
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if self.orientation_strength[i, j] < 0.20:
                    continue
                by = i * self.block_size + self.block_size // 2
                bx = j * self.block_size + self.block_size // 2
                if self.segmentation_mask is not None:
                    if 0 <= by < self.segmentation_mask.shape[0] and 0 <= bx < self.segmentation_mask.shape[1]:
                        if self.segmentation_mask[by, bx] == 0:
                            continue
                pi = self._compute_poincare_index(i, j)
                if abs(pi - np.pi) < np.pi / 4:
                    self.cores.append((j, i))
                elif abs(pi + np.pi) < np.pi / 4:
                    self.deltas.append((j, i))
        self.cores = self._merge_nearby(self.cores)
        self.deltas = self._merge_nearby(self.deltas)

    def _compute_poincare_index(self, i, j):
        nbs = [
            (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j + 1),
            (i + 1, j + 1), (i + 1, j), (i + 1, j - 1), (i, j - 1),
        ]
        total = 0.0
        for k in range(8):
            diff = self.orientation_map[nbs[(k + 1) % 8]] - self.orientation_map[nbs[k]]
            while diff > np.pi / 2:
                diff -= np.pi
            while diff <= -np.pi / 2:
                diff += np.pi
            total += diff
        return total

    @staticmethod
    def _merge_nearby(pts, d=3):
        if not pts:
            return []
        merged, used = [], [False] * len(pts)
        for i, p1 in enumerate(pts):
            if used[i]:
                continue
            cluster, used[i] = [p1], True
            for j, p2 in enumerate(pts):
                if not used[j] and np.hypot(p1[0] - p2[0], p1[1] - p2[1]) < d:
                    cluster.append(p2)
                    used[j] = True
            merged.append((int(np.mean([p[0] for p in cluster])), int(np.mean([p[1] for p in cluster]))))
        return merged

    def _classify(self):
        nc, nd = len(self.cores), len(self.deltas)
        self.details = {"cores": nc, "deltas": nd, "core_positions": list(self.cores),
                        "delta_positions": list(self.deltas)}
        if nc == 0 and nd == 0:
            self.classification, self.confidence = self.ARCH, 0.85
        elif nc == 1 and nd == 0:
            self.classification, self.confidence = self.TENTED_ARCH, 0.70
        elif nc == 1 and nd == 1:
            self.classification = self.LEFT_LOOP if self.deltas[0][0] > self.cores[0][0] else self.RIGHT_LOOP
            self.confidence = 0.80
        elif nc >= 2 or nd >= 2:
            if nc >= 2 and nd >= 2:
                self.classification, self.confidence = self.DOUBLE_LOOP_WHORL, 0.75
            elif nc == 1 and nd >= 2:
                self.classification, self.confidence = self.CENTRAL_POCKET, 0.70
            else:
                self.classification, self.confidence = self.PLAIN_WHORL, 0.70
        else:
            self.classification, self.confidence = self.UNKNOWN, 0.3

        if self.orientation_strength is not None:
            v = self.orientation_strength > 0.1
            coh = float(np.mean(self.orientation_strength[v])) if np.any(v) else 0.0
            self.confidence *= 0.5 + 0.5 * min(coh, 1.0)
            self.details["avg_coherence"] = coh
        if self.segmentation_mask is not None:
            cov = np.count_nonzero(self.segmentation_mask) / self.segmentation_mask.size
            if cov < 0.15:
                self.confidence *= 0.7
            elif cov < 0.30:
                self.confidence *= 0.85
            self.details["coverage"] = cov
        self.confidence = min(self.confidence, 0.99)

    def _assess_quality(self):
        if self.gray is None:
            self.quality_score = 0.0
            return
        contrast = np.std(self.gray) / 128.0
        cov = (np.count_nonzero(self.segmentation_mask) / self.segmentation_mask.size
               if self.segmentation_mask is not None else 0.5)
        coh = (float(np.mean(self.orientation_strength[self.orientation_strength > 0.1]))
               if self.orientation_strength is not None and np.any(self.orientation_strength > 0.1) else 0.5)
        rd = (min(np.count_nonzero(self.thinned) / np.count_nonzero(self.segmentation_mask), 1.0)
              if (self.thinned is not None and self.segmentation_mask is not None
                  and np.count_nonzero(self.segmentation_mask) > 0) else 0.5)
        self.quality_score = float(np.clip(0.25 * contrast + 0.20 * cov + 0.35 * min(coh, 1.0) + 0.20 * rd, 0, 1))
        self.details.update({"quality": self.quality_score, "contrast": contrast, "coverage": cov, "ridge_density": rd})

    def get_orientation_overlay(self):
        if self.orientation_map is None or self.original is None:
            return None
        ov = self.original.copy()
        rows, cols = self.orientation_map.shape
        bh, bw = self.block_size, self.block_size
        half = bh // 3
        h, w = ov.shape[:2]
        for i in range(rows):
            for j in range(cols):
                if self.orientation_strength[i, j] < 0.1:
                    continue
                cx, cy = j * bw + bw // 2, i * bh + bh // 2
                if cx >= w or cy >= h:
                    continue
                if (self.segmentation_mask is not None
                        and 0 <= cy < self.segmentation_mask.shape[0]
                        and 0 <= cx < self.segmentation_mask.shape[1]
                        and self.segmentation_mask[cy, cx] == 0):
                    continue
                t = self.orientation_map[i, j]
                dx, dy = int(half * np.cos(t)), int(half * np.sin(t))
                s = min(self.orientation_strength[i, j], 1.0)
                cv2.line(ov, (max(0, cx - dx), max(0, cy - dy)),
                         (min(w - 1, cx + dx), min(h - 1, cy + dy)),
                         (int(255 * (1 - s)), int(200 * s), int(255 * s)), 1, cv2.LINE_AA)
        return ov

    def get_singular_points_overlay(self):
        if self.original is None:
            return None
        ov = self.original.copy()
        bh, bw = self.block_size, self.block_size
        r = max(bh, bw) // 2 + 5
        h, w = ov.shape[:2]
        for cx, cy in self.cores:
            px, py = cx * bw + bw // 2, cy * bh + bh // 2
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(ov, (px, py), r, (0, 0, 255), 2)
                cv2.putText(ov, "Core", (max(0, px - 15), max(12, py - r - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        for cx, cy in self.deltas:
            px, py = cx * bw + bw // 2, cy * bh + bh // 2
            if 0 <= px < w and 0 <= py < h:
                cv2.polylines(ov, [np.array([[px, py - r], [px - r, py + r], [px + r, py + r]])],
                              True, (255, 150, 0), 2)
                cv2.putText(ov, "Delta", (max(0, px - 15), min(h - 4, py + r + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
        return ov


# ═══════════════════════════════════════════════════════════════════════
# GENERAL IMAGE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

class ImageClassifier:
    """Feature-based general image classifier.

    Detects whether an image is a fingerprint, document, photograph, etc.
    Uses color histograms, edge density, texture energy, and orientation
    coherence — all computed with OpenCV.
    """

    CATEGORIES = {
        "Fingerprint": " Ridge-like oriented pattern detected",
        "Document": " High-contrast text/line content",
        "Photograph": " Natural scene or portrait photo",
        "QR/Barcode": " Binary pattern with finder marks",
        "Texture/Pattern": " Repetitive non-ridge pattern",
        "Low Quality": " Image too dark, bright, or blurry",
        "Other": " Unclassified image type",
    }

    @staticmethod
    def classify(gray_img, color_img=None):
        """Return (category, confidence, features_dict)."""
        if gray_img is None:
            return "Other", 0.0, {}

        h, w = gray_img.shape
        features = {}

        # ── Basic statistics ─────────────────────────────────────────
        mean_i = float(np.mean(gray_img))
        std_i = float(np.std(gray_img))
        features["mean_intensity"] = mean_i
        features["std_intensity"] = std_i

        # ── Edge density ─────────────────────────────────────────────
        edges = cv2.Canny(gray_img, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / (h * w)
        features["edge_density"] = edge_density

        # ── Orientation coherence (fingerprint detector) ─────────────
        gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
        gxx = cv2.GaussianBlur(gx * gx, (15, 15), 5)
        gyy = cv2.GaussianBlur(gy * gy, (15, 15), 5)
        gxy = cv2.GaussianBlur(gx * gy, (15, 15), 5)
        denom = gxx + gyy + 1e-10
        coherence = np.mean(np.minimum(gxx, gyy) / denom)
        coherence_map = 1.0 - 2.0 * np.minimum(gxx, gyy) / denom
        avg_coherence = float(np.mean(coherence_map))
        features["orientation_coherence"] = avg_coherence

        # ── Ridge frequency estimate ─────────────────────────────────
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_img)
        rs = min(h, w) // 3
        region = enhanced[h // 2 - rs // 2:h // 2 + rs // 2, w // 2 - rs // 2:w // 2 + rs // 2]
        freq_estimates = []
        for sig in [np.mean(region, axis=0), np.mean(region, axis=1)]:
            n = len(sig)
            if n < 8:
                continue
            ac = np.correlate(sig - np.mean(sig), sig - np.mean(sig), mode="full")[n - 1:]
            for k in range(3, len(ac) // 2):
                if ac[k] > ac[k - 1] and ac[k] > ac[k + 1] and ac[k] > 0.15 * ac[0]:
                    freq_estimates.append(1.0 / max(k, 3))
                    break
        ridge_freq = float(np.median(freq_estimates)) if freq_estimates else 0.0
        features["ridge_frequency"] = ridge_freq

        # ── Color analysis (if color available) ──────────────────────
        is_grayscale = True
        if color_img is not None and len(color_img.shape) == 3:
            b, g, r = cv2.split(color_img)
            color_var = float(np.mean(np.var(color_img.astype(float), axis=2)))
            features["color_variance"] = color_var
            is_grayscale = color_var < 50
        features["is_grayscale"] = is_grayscale

        # ── Local Binary Pattern energy ──────────────────────────────
        small = cv2.resize(gray_img, (128, 128))
        lbp = np.zeros_like(small)
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
            shifted = np.roll(np.roll(small.astype(int), dy, axis=0), dx, axis=1)
            lbp += ((shifted >= small.astype(int)) * (1 << ([(-1, -1), (-1, 0), (-1, 1), (0, 1),
                                                              (1, 1), (1, 0), (1, -1), (0, -1)].index((dy, dx)))).astype(np.uint8)
        lbp_hist = np.histogram(lbp, bins=32, range=(0, 256))[0].astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-10)
        lbp_entropy = -float(np.sum(lbp_hist * np.log2(lbp_hist + 1e-10)))
        features["lbp_entropy"] = lbp_entropy

        # ── Connected component count (for barcode/QR) ───────────────
        binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 10)
        n_labels, _ = cv2.connectedComponents(binary)
        features["component_count"] = n_labels

        # ── Classification rules ─────────────────────────────────────
        scores = {}

        # Fingerprint: high coherence, moderate frequency, grayscale
        fp_score = 0.0
        if avg_coherence > 0.3:
            fp_score += 0.35
        if is_grayscale:
            fp_score += 0.20
        if 0.02 < ridge_freq < 0.15:
            fp_score += 0.30
        if 0.03 < edge_density < 0.25:
            fp_score += 0.15
        scores["Fingerprint"] = min(fp_score, 1.0)

        # Document: high edge density, high contrast, high LBP entropy
        doc_score = 0.0
        if edge_density > 0.12:
            doc_score += 0.30
        if std_i > 50:
            doc_score += 0.25
        if n_labels > 100:
            doc_score += 0.25
        if lbp_entropy > 4.0:
            doc_score += 0.20
        scores["Document"] = min(doc_score, 1.0)

        # Photograph: color, low-moderate coherence
        photo_score = 0.0
        if not is_grayscale:
            photo_score += 0.40
        if avg_coherence < 0.4:
            photo_score += 0.25
        if lbp_entropy > 3.5:
            photo_score += 0.15
        if edge_density < 0.15:
            photo_score += 0.20
        scores["Photograph"] = min(photo_score, 1.0)

        # QR/Barcode: binary-like, many components, high contrast
        qr_score = 0.0
        if std_i > 60:
            qr_score += 0.25
        if n_labels > 200:
            qr_score += 0.30
        if edge_density > 0.15:
            qr_score += 0.25
        if is_grayscale:
            qr_score += 0.20
        scores["QR/Barcode"] = min(qr_score, 1.0)

        # Texture/Pattern: moderate coherence, non-fingerprint frequency
        tex_score = 0.0
        if 0.2 < avg_coherence < 0.5:
            tex_score += 0.30
        if lbp_entropy > 3.0:
            tex_score += 0.30
        if ridge_freq == 0 or ridge_freq > 0.15:
            tex_score += 0.20
        if edge_density > 0.05:
            tex_score += 0.20
        scores["Texture/Pattern"] = min(tex_score, 1.0)

        # Low quality
        lq_score = 0.0
        if mean_i < 40 or mean_i > 215:
            lq_score += 0.40
        if std_i < 20:
            lq_score += 0.40
        if edge_density < 0.02:
            lq_score += 0.20
        scores["Low Quality"] = min(lq_score, 1.0)

        # Pick best category
        best_cat = max(scores, key=scores.get)
        best_conf = scores[best_cat]
        features["all_scores"] = {k: round(v, 3) for k, v in scores.items()}

        # If nothing stands out
        if best_conf < 0.25:
            best_cat = "Other"

        return best_cat, round(best_conf, 3), features


# ═══════════════════════════════════════════════════════════════════════
# CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════════════════

class HistogramWidget(Widget):
    """Custom-painted grayscale histogram display."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hist = None
        self.bind(size=self._redraw, pos=self._redraw)

    def set_image(self, gray_image):
        if gray_image is None:
            self._hist = None
        else:
            self._hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        self._redraw()

    def _redraw(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(0.08, 0.08, 0.14, 1)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(6)])
            Color(0.15, 0.15, 0.25, 1)
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, dp(6)), width=1)
            if self._hist is None or self._hist.size == 0:
                return
            mx = float(np.max(self._hist))
            if mx <= 0:
                return
            m = dp(6)
            pw, ph = self.width - 2 * m, self.height - 2 * m - dp(12)
            bw = max(1.0, pw / 256.0)
            for i in range(256):
                bh_val = int(self._hist[i][0] / mx * ph)
                x = self.x + m + int(i * pw / 256)
                y = self.y + m + dp(12)
                t = i / 255.0
                Color(0.91 * (1 - t) + 0.33 * t, 0.27 * (1 - t) + 0.20 * t,
                      0.38 * (1 - t) + 0.51 * t, 0.85)
                Rectangle(pos=(x, y), size=(max(1, int(bw) + 1), max(0, bh_val)))


class GradientProgressBar(Widget):
    value = NumericProperty(0)
    max_value = NumericProperty(100)
    bar_color_start = ListProperty([0.91, 0.27, 0.38])
    bar_color_end = ListProperty([0.33, 0.20, 0.51])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(20)
        self.bind(value=self._redraw, size=self._redraw, pos=self._redraw)

    def _redraw(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(0.08, 0.08, 0.14, 1)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(4)])
            Color(0.15, 0.15, 0.25, 1)
            Line(rounded_rectangle=(self.x, self.y, self.width, self.height, dp(4)), width=1)
            frac = max(0, min(self.value / max(self.max_value, 1), 1.0))
            fill_w = max(0, self.width * frac - dp(2))
            if fill_w > 0:
                steps = max(1, int(fill_w / 2))
                step_w = fill_w / steps
                r0, g0, b0 = self.bar_color_start
                r1, g1, b1 = self.bar_color_end
                for s in range(steps):
                    t = s / max(steps - 1, 1)
                    Color(r0 + (r1 - r0) * t, g0 + (g1 - g0) * t, b0 + (b1 - b0) * t, 1)
                    Rectangle(pos=(self.x + dp(1) + s * step_w, self.y + dp(1)),
                              size=(step_w + 1, self.height - dp(2)))


class TabImageWidget(FloatLayout):
    """Image display with zoom/pan."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scatter = Scatter(do_rotation=False, do_scale=True, size_hint=(None, None))
        self._image = KivyImage(allow_stretch=False, keep_ratio=False, size_hint=(None, None))
        self._scatter.add_widget(self._image)
        self._placeholder = Label(
            text="📷 No Image", color=(0.33, 0.33, 0.40, 1), font_size=sp(14),
            size_hint=(1, 1), halign="center", valign="middle",
        )
        self.add_widget(self._placeholder)
        self.add_widget(self._scatter)
        self._texture = None

    def set_texture(self, texture):
        self._texture = texture
        if texture is None:
            self._image.texture = None
            self._scatter.size = (1, 1)
            self._image.size = (1, 1)
            self._placeholder.opacity = 1
            return
        self._placeholder.opacity = 0
        self._image.texture = texture
        self._image.size = texture.size
        self._scatter.size = texture.size
        self._fit_to_view()

    def _fit_to_view(self, *args):
        if self._texture is None or self.parent is None:
            return
        tw, th = self._texture.size
        if tw == 0 or th == 0:
            return
        pw, ph = self.size
        if pw <= 0 or ph <= 0:
            return
        scale = min(pw / tw, ph / th) * 0.92
        self._scatter.scale = scale
        self._scatter.center = self.center

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos) and touch.is_mouse_scrolling:
            if touch.button == "scrollup":
                self._scatter.scale = min(self._scatter.scale * 1.15, 8.0)
            elif touch.button == "scrolldown":
                self._scatter.scale = max(self._scatter.scale / 1.15, 0.15)
            return True
        if touch.is_double_tap and self.collide_point(*touch.pos):
            self._fit_to_view()
            return True
        return super().on_touch_down(touch)


class CardWidget(BoxLayout):
    """A rounded card container with a dark background."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = dp(10)
        self.spacing = dp(4)
        with self.canvas.before:
            Color(0.08, 0.10, 0.18, 1)
            self._bg = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])
        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *args):
        self._bg.pos = self.pos
        self._bg.size = self.size


class NavButton(Button):
    """Bottom navigation button."""

    active = BooleanProperty(False)

    def __init__(self, icon="", text="", **kwargs):
        super().__init__(**kwargs)
        self.icon_text = icon
        self.label_text = text
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = (0.5, 0.5, 0.55, 1)
        self.font_size = sp(10)
        self.halign = "center"
        self.valign = "middle"
        self.markup = True
        self._update_text()
        self.bind(active=self._update_text)

    def _update_text(self, *args):
        if self.active:
            self.text = f"[color=e8455a]{self.icon_text}[/color]\n[b]{self.label_text}[/b]"
        else:
            self.text = f"{self.icon_text}\n{self.label_text}"


# ═══════════════════════════════════════════════════════════════════════
# SCREENS
# ═══════════════════════════════════════════════════════════════════════

class HomeScreen(Screen):
    """Dashboard with quick actions and recent scans."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "home"
        root = BoxLayout(orientation="vertical")

        # Header
        header = BoxLayout(size_hint_y=None, height=dp(56), padding=[dp(16), dp(10)])
        with header.canvas.before:
            Color(0.05, 0.06, 0.12, 1)
            self._hbg = Rectangle(pos=header.pos, size=header.size)
        header.bind(pos=lambda i, v: setattr(self._hbg, 'pos', v),
                    size=lambda i, v: setattr(self._hbg, 'size', v))
        header.add_widget(Label(text="[b]🔍 ScanClassify[/b]", markup=True,
                                font_size=sp(20), color=(0.91, 0.27, 0.38, 1),
                                halign="left", valign="middle", size_hint_x=0.6))
        header.add_widget(Label(text="", size_hint_x=0.4))
        root.add_widget(header)

        # Scrollable content
        scroll = ScrollView()
        content = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(10), padding=dp(12))
        content.bind(minimum_height=content.setter("height"))

        # Welcome card
        welcome_card = CardWidget(size_hint_y=None, height=dp(100))
        welcome_card.add_widget(Label(
            text="Welcome to ScanClassify!\nClassify fingerprints & images\nfrom your phone or files.",
            font_size=sp(13), color=(0.80, 0.80, 0.86, 1), halign="center", valign="middle",
            size_hint_y=None, height=dp(80)))
        content.add_widget(welcome_card)

        # Quick actions
        actions_label = Label(text="Quick Actions", font_size=sp(14), bold=True,
                              color=(0.91, 0.27, 0.38, 1), size_hint_y=None, height=dp(28),
                              halign="left", valign="middle")
        content.add_widget(actions_label)

        actions_grid = GridLayout(cols=2, size_hint_y=None, height=dp(180), spacing=dp(8))
        actions = [
            ("👆", "Scan\nFingerprint", "fingerprint"),
            ("📷", "Capture\nfrom Camera", "fingerprint"),
            ("🖼️", "Classify\nImage", "classify"),
            ("📂", "Load\nfrom File", "fingerprint"),
        ]
        for icon, label, target in actions:
            btn = Button(
                text=f"{icon}\n{label}", markup=True, font_size=sp(12),
                background_normal="", background_down="",
                background_color=(0.08, 0.10, 0.18, 1),
                color=(0.85, 0.85, 0.92, 1),
            )
            btn.bind(on_press=lambda inst, t=target: self._go_to(t))
            actions_grid.add_widget(btn)
        content.add_widget(actions_grid)

        # Recent scans
        content.add_widget(Label(text="Recent Scans", font_size=sp(14), bold=True,
                                 color=(0.91, 0.27, 0.38, 1), size_hint_y=None, height=dp(28),
                                 halign="left", valign="middle"))

        self.recent_container = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4))
        self.recent_container.bind(minimum_height=self.recent_container.setter("height"))
        content.add_widget(self.recent_container)

        scroll.add_widget(content)
        root.add_widget(scroll)
        self.add_widget(root)
        self._refresh_recent()

    def _go_to(self, target):
        app = App.get_running_app()
        if target == "fingerprint":
            app.sm.current = "fingerprint"
        elif target == "classify":
            app.sm.current = "classify"

    def _refresh_recent(self):
        self.recent_container.clear_widgets()
        records = history_mgr.recent(5)
        if not records:
            self.recent_container.add_widget(Label(
                text="No scans yet. Try scanning a fingerprint!",
                font_size=sp(11), color=(0.5, 0.5, 0.55, 1),
                size_hint_y=None, height=dp(40)))
            return
        for rec in records:
            ts = rec.get("timestamp", "")[:16].replace("T", " ")
            cls = rec.get("classification", "?")
            conf = int(rec.get("confidence", 0) * 100)
            row = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(6))
            row.add_widget(Label(text=f"🕐 {ts}", font_size=sp(10),
                                 color=(0.6, 0.6, 0.65, 1), size_hint_x=0.45))
            row.add_widget(Label(text=f"{cls} ({conf}%)", font_size=sp(11),
                                 color=(0.85, 0.85, 0.92, 1), size_hint_x=0.35, bold=True))
            src = "📸" if rec.get("source") == "camera" else "💾"
            row.add_widget(Label(text=src, font_size=sp(14), size_hint_x=0.2))
            self.recent_container.add_widget(row)

    def on_enter(self, *args):
        self._refresh_recent()


class FingerprintScreen(Screen):
    """Fingerprint classification with load, camera, generate."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "fingerprint"
        self.processor = FingerprintProcessor()
        self._processing = False
        self._camera_capture = None

        root = BoxLayout(orientation="vertical")

        # Header
        header = BoxLayout(size_hint_y=None, height=dp(50), padding=[dp(12), dp(6)])
        with header.canvas.before:
            Color(0.05, 0.06, 0.12, 1)
            self._hbg = Rectangle(pos=header.pos, size=header.size)
        header.bind(pos=lambda i, v: setattr(self._hbg, 'pos', v),
                    size=lambda i, v: setattr(self._hbg, 'size', v))
        back_btn = Button(text="← Back", font_size=sp(12), size_hint_x=0.2,
                          background_normal="", background_down="",
                          background_color=(0.15, 0.15, 0.25, 1), color=(0.8, 0.8, 0.86, 1))
        back_btn.bind(on_press=lambda x: setattr(App.get_running_app().sm, 'current', 'home'))
        header.add_widget(back_btn)
        header.add_widget(Label(text="[b]👆 Fingerprint Scanner[/b]", markup=True,
                                font_size=sp(16), color=(0.91, 0.27, 0.38, 1)))
        root.add_widget(header)

        # Main scrollable
        scroll = ScrollView()
        content = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(6), padding=dp(10))
        content.bind(minimum_height=content.setter("height"))

        # Image display
        self.image_display = TabImageWidget(size_hint_y=None, height=dp(220))
        content.add_widget(self.image_display)

        # Action buttons row
        btn_row = GridLayout(cols=3, size_hint_y=None, height=dp(42), spacing=dp(4))
        btn_load = Button(text="📂 Load", font_size=sp(11), background_normal="",
                          background_down="", background_color=(0.06, 0.13, 0.24, 1),
                          color=(0.85, 0.85, 0.94, 1))
        btn_load.bind(on_press=self._show_file_chooser)
        btn_row.add_widget(btn_load)

        btn_cam = Button(text="📷 Camera", font_size=sp(11), background_normal="",
                         background_down="", background_color=(0.06, 0.13, 0.24, 1),
                         color=(0.85, 0.85, 0.94, 1))
        btn_cam.bind(on_press=self._open_camera)
        btn_row.add_widget(btn_cam)

        btn_gen = Button(text="🎲 Generate", font_size=sp(11), background_normal="",
                         background_down="", background_color=(0.06, 0.13, 0.24, 1),
                         color=(0.85, 0.85, 0.94, 1))
        btn_gen.bind(on_press=self._generate_pattern)
        btn_row.add_widget(btn_gen)
        content.add_widget(btn_row)

        # Generate options (collapsible feel)
        gen_card = CardWidget(size_hint_y=None, height=dp(115), spacing=dp(3))
        gen_card.add_widget(Label(text="Generate Options", font_size=sp(11), bold=True,
                                  color=(0.58, 0.77, 0.99, 1), size_hint_y=None, height=dp(18)))
        self.pattern_spinner = Spinner(text="Whorl", values=FingerprintGenerator.PATTERNS,
                                       size_hint_y=None, height=dp(32), background_normal="",
                                       background_down="", background_color=(0.06, 0.13, 0.24, 1),
                                       color=(0.85, 0.85, 0.94, 1), font_size=sp(11))
        gen_card.add_widget(self.pattern_spinner)

        noise_row = BoxLayout(size_hint_y=None, height=dp(28))
        noise_row.add_widget(Label(text="Noise:", font_size=sp(10), color=(0.7, 0.7, 0.76, 1),
                                   size_hint_x=0.25))
        self.noise_slider = Slider(min=0.0, max=0.5, value=0.15, step=0.01)
        noise_row.add_widget(self.noise_slider)
        self.noise_val = Label(text="0.15", font_size=sp(10), color=(0.7, 0.7, 0.76, 1),
                               size_hint_x=0.15)
        self.noise_slider.bind(value=lambda i, v: setattr(self.noise_val, 'text', f'{v:.2f}'))
        noise_row.add_widget(self.noise_val)
        gen_card.add_widget(noise_row)

        block_row = BoxLayout(size_hint_y=None, height=dp(28))
        block_row.add_widget(Label(text="Block:", font_size=sp(10), color=(0.7, 0.7, 0.76, 1),
                                   size_hint_x=0.25))
        self.block_slider = Slider(min=8, max=32, value=16, step=2)
        block_row.add_widget(self.block_slider)
        self.block_val = Label(text="16", font_size=sp(10), color=(0.7, 0.7, 0.76, 1),
                               size_hint_x=0.15)
        self.block_slider.bind(value=lambda i, v: setattr(self.block_val, 'text', str(int(v))))
        block_row.add_widget(self.block_val)
        gen_card.add_widget(block_row)
        content.add_widget(gen_card)

        # Process button
        self.btn_process = Button(text="▶  Process & Classify", size_hint_y=None, height=dp(46),
                                  background_normal="", background_down="",
                                  background_color=(0.91, 0.27, 0.38, 1),
                                  color=(1, 1, 1, 1), font_size=sp(15), bold=True)
        self.btn_process.bind(on_press=self._process_image)
        content.add_widget(self.btn_process)

        # Status
        self.status_label = Label(text="Ready — load, capture, or generate a fingerprint",
                                  size_hint_y=None, height=dp(22),
                                  color=(0.4, 0.8, 0.4, 1), font_size=sp(10))
        content.add_widget(self.status_label)

        # Results card
        self.result_card = CardWidget(size_hint_y=None, height=dp(200), spacing=dp(4))
        self.result_card.add_widget(Label(text="Classification Result", font_size=sp(13), bold=True,
                                          color=(0.91, 0.27, 0.38, 1), size_hint_y=None, height=dp(24)))
        self.class_label = Label(text="—", font_size=sp(22), bold=True,
                                 color=(1, 1, 1, 1), size_hint_y=None, height=dp(34))
        self.result_card.add_widget(self.class_label)

        # Confidence bar
        conf_row = BoxLayout(size_hint_y=None, height=dp(28))
        conf_row.add_widget(Label(text="Confidence:", font_size=sp(10), color=(0.7, 0.7, 0.76, 1),
                                  size_hint_x=0.3))
        self.confidence_bar = GradientProgressBar(size_hint_x=0.5)
        conf_row.add_widget(self.confidence_bar)
        self.conf_pct = Label(text="0%", font_size=sp(10), color=(0.7, 0.7, 0.76, 1),
                              size_hint_x=0.2)
        conf_row.add_widget(self.conf_pct)
        self.result_card.add_widget(conf_row)

        # Quality bar
        qual_row = BoxLayout(size_hint_y=None, height=dp(28))
        qual_row.add_widget(Label(text="Quality:", font_size=sp(10), color=(0.7, 0.7, 0.76, 1),
                                  size_hint_x=0.3))
        self.quality_bar = GradientProgressBar(bar_color_start=[0.06, 0.20, 0.38],
                                               bar_color_end=[0.0, 0.71, 0.85], size_hint_x=0.5)
        qual_row.add_widget(self.quality_bar)
        self.qual_pct = Label(text="0%", font_size=sp(10), color=(0.7, 0.7, 0.76, 1),
                              size_hint_x=0.2)
        qual_row.add_widget(self.qual_pct)
        self.result_card.add_widget(qual_row)

        self.details_label = Label(text="", font_size=sp(10), color=(0.75, 0.75, 0.80, 1),
                                   halign="left", valign="top", size_hint_y=None, height=dp(60))
        self.result_card.add_widget(self.details_label)
        content.add_widget(self.result_card)

        # View tabs card
        view_card = CardWidget(size_hint_y=None, height=dp(290), spacing=dp(2))
        view_card.add_widget(Label(text="Processing Views", font_size=sp(11), bold=True,
                                   color=(0.58, 0.77, 0.99, 1), size_hint_y=None, height=dp(20)))
        self.tab_panel = TabbedPanel(tab_pos="top_mid", tab_height=dp(30), tab_width=dp(80))
        self.tab_images = {}
        tab_configs = [
            ("Original", "original"), ("Enhanced", "enhanced"),
            ("Orientation", "orientation"), ("Singular Pts", "singular"),
            ("Binary", "binary"), ("Thinned", "thinned"),
        ]
        for tab_text, key in tab_configs:
            tab_item = TabbedPanelItem(text=tab_text)
            img_widget = TabImageWidget()
            tab_item.add_widget(img_widget)
            self.tab_images[key] = img_widget
            self.tab_panel.add_widget(tab_item)
        view_card.add_widget(self.tab_panel)
        content.add_widget(view_card)

        # Histogram
        content.add_widget(Label(text="Histogram", font_size=sp(11), bold=True,
                                 color=(0.58, 0.77, 0.99, 1), size_hint_y=None, height=dp(20)))
        self.histogram = HistogramWidget(size_hint_y=None, height=dp(120))
        content.add_widget(self.histogram)

        # Processing steps
        steps_card = CardWidget(size_hint_y=None, height=dp(200), spacing=dp(1))
        steps_card.add_widget(Label(text="Processing Steps", font_size=sp(11), bold=True,
                                    color=(0.58, 0.77, 0.99, 1), size_hint_y=None, height=dp(18)))
        self.step_indicators = {}
        for step_name in FingerprintProcessor.STEPS:
            lbl = Label(text=f"  ○ {step_name}", size_hint_y=None, height=dp(16),
                        color=(0.50, 0.50, 0.55, 1), font_size=sp(9), halign="left", valign="middle")
            self.step_indicators[step_name] = lbl
            steps_card.add_widget(lbl)
        content.add_widget(steps_card)

        # Save button
        btn_save = Button(text="💾  Save Enhanced Image", size_hint_y=None, height=dp(38),
                          background_normal="", background_down="",
                          background_color=(0.06, 0.13, 0.24, 1),
                          color=(0.85, 0.85, 0.94, 1), font_size=sp(12))
        btn_save.bind(on_press=self._save_enhanced)
        content.add_widget(btn_save)

        scroll.add_widget(content)
        root.add_widget(scroll)
        self.add_widget(root)

    # ── File Chooser ────────────────────────────────────────────────
    def _show_file_chooser(self, instance=None):
        content = BoxLayout(orientation="vertical", spacing=dp(6))
        default_path = os.path.expanduser("~")
        fc = FileChooserListView(path=default_path if os.path.isdir(default_path) else "/",
                                 filters=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"])
        content.add_widget(fc)
        btn_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        btn_cancel = Button(text="Cancel", background_normal="", background_down="",
                            background_color=(0.06, 0.13, 0.24, 1), color=(0.85, 0.85, 0.94, 1))
        btn_load = Button(text="Load", background_normal="", background_down="",
                          background_color=(0.91, 0.27, 0.38, 1), color=(1, 1, 1, 1))
        btn_row.add_widget(btn_cancel)
        btn_row.add_widget(btn_load)
        content.add_widget(btn_row)
        popup = Popup(title="Load Fingerprint Image", content=content,
                      size_hint=(0.9, 0.9), background_color=(0.06, 0.08, 0.15, 1),
                      separator_color=(0.91, 0.27, 0.38, 1), title_color=(1, 1, 1, 1))

        def on_load(inst):
            if fc.selection:
                self._load_file(fc.selection[0])
            popup.dismiss()

        btn_load.bind(on_press=on_load)
        btn_cancel.bind(on_press=lambda x: popup.dismiss())
        popup.open()

    def _load_file(self, path):
        try:
            self.processor.load_image(path)
            self._update_after_load()
            self._set_status("Image loaded ✔", (0.4, 0.8, 0.4, 1))
        except Exception as e:
            self._show_error(f"Failed to load:\n{e}")

    # ── Camera Capture ──────────────────────────────────────────────
    def _open_camera(self, instance=None):
        content = BoxLayout(orientation="vertical", spacing=dp(6))

        try:
            cam = Camera(play=True, resolution=(640, 480), size_hint_y=0.85)
            content.add_widget(cam)
        except Exception as e:
            content.add_widget(Label(text=f"Camera unavailable:\n{e}\n\n"
                                          "On Android, use a camera app\n"
                                          "then load the photo via 📂 Load.",
                                     font_size=sp(12), color=(0.85, 0.85, 0.92, 1)))

        btn_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        btn_cancel = Button(text="Cancel", background_normal="", background_down="",
                            background_color=(0.06, 0.13, 0.24, 1), color=(0.85, 0.85, 0.94, 1))
        btn_capture = Button(text="📸  Capture", background_normal="", background_down="",
                             background_color=(0.91, 0.27, 0.38, 1), color=(1, 1, 1, 1),
                             font_size=sp(14), bold=True)
        btn_row.add_widget(btn_cancel)
        btn_row.add_widget(btn_capture)
        content.add_widget(btn_row)

        hint = Label(text="Place your finger on the camera lens\nfor best fingerprint capture",
                     font_size=sp(10), color=(0.5, 0.5, 0.55, 1), size_hint_y=None, height=dp(30))
        content.add_widget(hint)

        popup = Popup(title="📷 Capture Fingerprint", content=content,
                      size_hint=(0.95, 0.95), background_color=(0.06, 0.08, 0.15, 1),
                      separator_color=(0.91, 0.27, 0.38, 1), title_color=(1, 1, 1, 1))

        def on_capture(inst):
            try:
                tex = cam.texture
                if tex is not None:
                    # Convert Kivy texture → numpy
                    w, h = tex.size
                    buf = tex.pixels
                    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                    rgb = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    # Save capture
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cap_path = HISTORY_DIR / f"capture_{ts}.png"
                    cv2.imwrite(str(cap_path), gray)
                    self.processor.load_from_array(gray, label="camera")
                    self._update_after_load()
                    self._set_status("Fingerprint captured ✔", (0.4, 0.8, 0.4, 1))
                    cam.play = False
            except Exception as e:
                self._show_error(f"Capture failed:\n{e}")
            popup.dismiss()

        btn_capture.bind(on_press=on_capture)
        btn_cancel.bind(on_press=lambda x: (setattr(cam, 'play', False) if hasattr(cam, 'play') else None,
                                            popup.dismiss()))
        popup.open()

    # ── Generate ────────────────────────────────────────────────────
    def _generate_pattern(self, instance=None):
        try:
            pattern = self.pattern_spinner.text
            noise = self.noise_slider.value
            img = FingerprintGenerator.generate(size=300, pattern=pattern, noise_level=noise)
            self.processor.load_from_array(img, label=f"synthetic_{pattern}")
            self._update_after_load()
            self._set_status(f"Generated: {pattern} ✔", (0.4, 0.8, 0.4, 1))
        except Exception as e:
            self._show_error(f"Generate failed:\n{e}")

    def _update_after_load(self):
        texture = cv2_to_texture(self.processor.gray)
        self.image_display.set_texture(texture)
        self.histogram.set_image(self.processor.gray)
        self._clear_results()
        # Also update original tab
        if "original" in self.tab_images:
            self.tab_images["original"].set_texture(texture)

    # ── Process ─────────────────────────────────────────────────────
    def _process_image(self, instance=None):
        if self._processing:
            return
        if self.processor.gray is None:
            self._show_error("No image loaded. Load, capture, or generate first.")
            return
        self._processing = True
        self.btn_process.disabled = True
        self.btn_process.text = "⏳ Processing..."
        self._set_status("Processing...", (0.91, 0.75, 0.27, 1))
        block_size = int(self.block_slider.value)

        def worker():
            try:
                self.processor.step_callback = self._on_step_callback
                self.processor.process(block_size=block_size)
                Clock.schedule_once(lambda dt: self._on_process_done(), 0)
            except Exception as e:
                Clock.schedule_once(lambda dt: self._on_process_error(str(e)), 0)

        threading.Thread(target=worker, daemon=True).start()

    def _on_step_callback(self, step_name, step_idx):
        Clock.schedule_once(lambda dt: self._update_step(step_name, step_idx), 0)

    def _update_step(self, step_name, step_idx):
        for i, (name, lbl) in enumerate(self.step_indicators.items()):
            if i < step_idx:
                lbl.text = f"  ✓ {name}"
                lbl.color = (0.3, 0.75, 0.3, 1)
            elif i == step_idx:
                lbl.text = f"  ● {name}"
                lbl.color = (0.91, 0.75, 0.27, 1)
            else:
                lbl.text = f"  ○ {name}"
                lbl.color = (0.50, 0.50, 0.55, 1)

    def _on_process_done(self):
        self._processing = False
        self.btn_process.disabled = False
        self.btn_process.text = "▶  Process & Classify"
        self._set_status("Done! ✔", (0.3, 0.85, 0.3, 1))

        for name, lbl in self.step_indicators.items():
            lbl.text = f"  ✓ {name}"
            lbl.color = (0.3, 0.75, 0.3, 1)

        self._update_all_tabs()
        self._update_results()
        self.histogram.set_image(self.processor.enhanced)

        # Save to history
        p = self.processor
        # Save thumbnail
        thumb = cv2.resize(p.gray if p.gray is not None else np.zeros((100, 100), dtype=np.uint8),
                           (100, 100))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        thumb_path = HISTORY_DIR / f"thumb_{ts}.png"
        cv2.imwrite(str(thumb_path), thumb)
        history_mgr.add(p.classification, p.confidence, p.quality_score, p.file_path or "unknown", thumb_path)

        # Switch to Enhanced tab
        for tab in self.tab_panel.tab_list:
            if tab.text == "Enhanced":
                self.tab_panel.switch_to(tab)
                break

    def _on_process_error(self, error_msg):
        self._processing = False
        self.btn_process.disabled = False
        self.btn_process.text = "▶  Process & Classify"
        self._show_error(f"Processing failed:\n{error_msg}")
        self._set_status("Error ✘", (0.91, 0.27, 0.27, 1))

    def _update_all_tabs(self):
        p = self.processor
        tab_data = {
            "original": p.gray, "enhanced": p.enhanced,
            "orientation": p.get_orientation_overlay(),
            "singular": p.get_singular_points_overlay(),
            "binary": p.binary_img, "thinned": p.thinned,
        }
        for key, img in tab_data.items():
            if key in self.tab_images:
                self.tab_images[key].set_texture(cv2_to_texture(img) if img is not None else None)
        # Also update main display
        self.image_display.set_texture(cv2_to_texture(p.enhanced))

    def _update_results(self):
        p = self.processor
        class_colors = {
            FingerprintProcessor.ARCH: (0.29, 0.87, 0.50),
            FingerprintProcessor.TENTED_ARCH: (0.53, 0.93, 0.67),
            FingerprintProcessor.LEFT_LOOP: (0.38, 0.65, 0.98),
            FingerprintProcessor.RIGHT_LOOP: (0.58, 0.77, 0.99),
            FingerprintProcessor.PLAIN_WHORL: (0.96, 0.45, 0.71),
            FingerprintProcessor.DOUBLE_LOOP_WHORL: (0.91, 0.47, 0.98),
            FingerprintProcessor.CENTRAL_POCKET: (0.75, 0.52, 0.99),
            FingerprintProcessor.UNKNOWN: (0.97, 0.44, 0.44),
        }
        self.class_label.text = p.classification
        self.class_label.color = class_colors.get(p.classification, (1, 1, 1, 1))

        conf_pct = int(p.confidence * 100)
        self.confidence_bar.value = conf_pct
        self.conf_pct.text = f"{conf_pct}%"

        qual_pct = int(p.quality_score * 100)
        self.quality_bar.value = qual_pct
        self.qual_pct.text = f"{qual_pct}%"

        d = p.details
        lines = [f"Cores: {d.get('cores', 0)}  |  Deltas: {d.get('deltas', 0)}"]
        if "avg_coherence" in d:
            lines.append(f"Coherence: {d['avg_coherence']:.2f}")
        if "contrast" in d:
            lines.append(f"Contrast: {d['contrast']:.2f}")
        if "coverage" in d:
            lines.append(f"Coverage: {d['coverage']:.1%}")
        if "ridge_density" in d:
            lines.append(f"Ridge density: {d['ridge_density']:.2f}")
        if "processing_time" in d:
            lines.append(f"Time: {d['processing_time']:.3f}s")
        self.details_label.text = "\n".join(lines)

    def _clear_results(self):
        self.class_label.text = "—"
        self.class_label.color = (1, 1, 1, 1)
        self.confidence_bar.value = 0
        self.conf_pct.text = "0%"
        self.quality_bar.value = 0
        self.qual_pct.text = "0%"
        self.details_label.text = ""
        for key in self.tab_images:
            self.tab_images[key].set_texture(None)
        for name, lbl in self.step_indicators.items():
            lbl.text = f"  ○ {name}"
            lbl.color = (0.50, 0.50, 0.55, 1)

    def _save_enhanced(self, instance=None):
        p = self.processor
        if p.enhanced is None:
            self._show_error("Nothing to save. Process an image first.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"fingerprint_{p.classification.replace(' ', '_')}_{ts}.png"
        out_path = HISTORY_DIR / default_name
        save_cv_image(p.enhanced, out_path)
        # Also save the original with overlay
        overlay_path = HISTORY_DIR / f"overlay_{default_name}"
        overlay = p.get_singular_points_overlay()
        if overlay is not None:
            save_cv_image(overlay, overlay_path)
        self._set_status(f"Saved to {out_path}", (0.4, 0.8, 0.4, 1))

    def _set_status(self, text, color):
        self.status_label.text = text
        self.status_label.color = color

    def _show_error(self, msg):
        popup = Popup(title="Error", content=Label(text=msg, font_size=sp(12)),
                      size_hint=(0.8, 0.4), background_color=(0.06, 0.08, 0.15, 1),
                      separator_color=(0.91, 0.27, 0.38, 1), title_color=(1, 1, 1, 1))
        popup.open()


class ImageClassifyScreen(Screen):
    """General image classification screen."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "classify"
        self._gray = None
        self._color = None

        root = BoxLayout(orientation="vertical")

        # Header
        header = BoxLayout(size_hint_y=None, height=dp(50), padding=[dp(12), dp(6)])
        with header.canvas.before:
            Color(0.05, 0.06, 0.12, 1)
            self._hbg = Rectangle(pos=header.pos, size=header.size)
        header.bind(pos=lambda i, v: setattr(self._hbg, 'pos', v),
                    size=lambda i, v: setattr(self._hbg, 'size', v))
        back_btn = Button(text="← Back", font_size=sp(12), size_hint_x=0.2,
                          background_normal="", background_down="",
                          background_color=(0.15, 0.15, 0.25, 1), color=(0.8, 0.8, 0.86, 1))
        back_btn.bind(on_press=lambda x: setattr(App.get_running_app().sm, 'current', 'home'))
        header.add_widget(back_btn)
        header.add_widget(Label(text="[b]🖼️ Image Classifier[/b]", markup=True,
                                font_size=sp(16), color=(0.91, 0.27, 0.38, 1)))
        root.add_widget(header)

        scroll = ScrollView()
        content = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(8), padding=dp(12))
        content.bind(minimum_height=content.setter("height"))

        # Image display
        self.image_display = TabImageWidget(size_hint_y=None, height=dp(200))
        content.add_widget(self.image_display)

        # Buttons
        btn_row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(6))
        btn_load = Button(text="📂 Load Image", font_size=sp(12), background_normal="",
                          background_down="", background_color=(0.06, 0.13, 0.24, 1),
                          color=(0.85, 0.85, 0.94, 1))
        btn_load.bind(on_press=self._show_file_chooser)
        btn_row.add_widget(btn_load)

        btn_cam = Button(text="📷 Camera", font_size=sp(12), background_normal="",
                         background_down="", background_color=(0.06, 0.13, 0.24, 1),
                         color=(0.85, 0.85, 0.94, 1))
        btn_cam.bind(on_press=self._open_camera)
        btn_row.add_widget(btn_cam)
        content.add_widget(btn_row)

        # Classify button
        self.btn_classify = Button(text="🔍  Classify Image", size_hint_y=None, height=dp(46),
                                   background_normal="", background_down="",
                                   background_color=(0.91, 0.27, 0.38, 1),
                                   color=(1, 1, 1, 1), font_size=sp(15), bold=True)
        self.btn_classify.bind(on_press=self._classify_image)
        content.add_widget(self.btn_classify)

        # Status
        self.status_label = Label(text="Load an image to classify", size_hint_y=None, height=dp(22),
                                  color=(0.5, 0.7, 0.5, 1), font_size=sp(10))
        content.add_widget(self.status_label)

        # Result card
        self.result_card = CardWidget(size_hint_y=None, height=dp(80))
        self.result_card.add_widget(Label(text="Image Type", font_size=sp(11), bold=True,
                                          color=(0.58, 0.77, 0.99, 1), size_hint_y=None, height=dp(18)))
        self.result_label = Label(text="—", font_size=sp(20), bold=True,
                                  color=(1, 1, 1, 1), size_hint_y=None, height=dp(30))
        self.result_card.add_widget(self.result_label)
        self.result_desc = Label(text="", font_size=sp(10), color=(0.7, 0.7, 0.76, 1),
                                 size_hint_y=None, height=dp(20))
        self.result_card.add_widget(self.result_desc)
        content.add_widget(self.result_card)

        # Confidence card
        self.scores_card = CardWidget(size_hint_y=None, height=dp(200))
        self.scores_card.add_widget(Label(text="Category Scores", font_size=sp(11), bold=True,
                                          color=(0.58, 0.77, 0.99, 1), size_hint_y=None, height=dp(18)))
        self.score_bars = {}
        for cat in ImageClassifier.CATEGORIES:
            row = BoxLayout(size_hint_y=None, height=dp(22))
            row.add_widget(Label(text=cat, font_size=sp(9), color=(0.7, 0.7, 0.76, 1),
                                 size_hint_x=0.30, halign="right"))
            bar = GradientProgressBar(size_hint_x=0.55)
            row.add_widget(bar)
            pct = Label(text="0%", font_size=sp(9), color=(0.7, 0.7, 0.76, 1), size_hint_x=0.15)
            row.add_widget(pct)
            self.score_bars[cat] = (bar, pct)
            self.scores_card.add_widget(row)
        content.add_widget(self.scores_card)

        # Features card
        self.features_card = CardWidget(size_hint_y=None, height=dp(160))
        self.features_card.add_widget(Label(text="Detected Features", font_size=sp(11), bold=True,
                                            color=(0.58, 0.77, 0.99, 1), size_hint_y=None, height=dp(18)))
        self.features_label = Label(text="", font_size=sp(10), color=(0.75, 0.75, 0.80, 1),
                                    halign="left", valign="top", size_hint_y=None, height=dp(130))
        self.features_card.add_widget(self.features_label)
        content.add_widget(self.features_card)

        # Histogram
        content.add_widget(Label(text="Histogram", font_size=sp(11), bold=True,
                                 color=(0.58, 0.77, 0.99, 1), size_hint_y=None, height=dp(18)))
        self.histogram = HistogramWidget(size_hint_y=None, height=dp(100))
        content.add_widget(self.histogram)

        scroll.add_widget(content)
        root.add_widget(scroll)
        self.add_widget(root)

    def _show_file_chooser(self, instance=None):
        content = BoxLayout(orientation="vertical", spacing=dp(6))
        default_path = os.path.expanduser("~")
        fc = FileChooserListView(path=default_path if os.path.isdir(default_path) else "/",
                                 filters=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"])
        content.add_widget(fc)
        btn_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        btn_cancel = Button(text="Cancel", background_normal="", background_down="",
                            background_color=(0.06, 0.13, 0.24, 1), color=(0.85, 0.85, 0.94, 1))
        btn_load = Button(text="Load", background_normal="", background_down="",
                          background_color=(0.91, 0.27, 0.38, 1), color=(1, 1, 1, 1))
        btn_row.add_widget(btn_cancel)
        btn_row.add_widget(btn_load)
        content.add_widget(btn_row)
        popup = Popup(title="Load Image for Classification", content=content,
                      size_hint=(0.9, 0.9), background_color=(0.06, 0.08, 0.15, 1),
                      separator_color=(0.91, 0.27, 0.38, 1), title_color=(1, 1, 1, 1))

        def on_load(inst):
            if fc.selection:
                self._load_file(fc.selection[0])
            popup.dismiss()

        btn_load.bind(on_press=on_load)
        btn_cancel.bind(on_press=lambda x: popup.dismiss())
        popup.open()

    def _load_file(self, path):
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Cannot load: {path}")
            max_dim = 800
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            self._color = img.copy()
            self._gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
            self.image_display.set_texture(cv2_to_texture(self._gray))
            self.histogram.set_image(self._gray)
            self.status_label.text = f"Image loaded ({w}×{h})"
            self.status_label.color = (0.4, 0.8, 0.4, 1)
        except Exception as e:
            self._show_error(f"Failed to load:\n{e}")

    def _open_camera(self, instance=None):
        content = BoxLayout(orientation="vertical", spacing=dp(6))
        try:
            cam = Camera(play=True, resolution=(640, 480), size_hint_y=0.85)
            content.add_widget(cam)
        except Exception as e:
            content.add_widget(Label(text=f"Camera unavailable:\n{e}",
                                     font_size=sp(12), color=(0.85, 0.85, 0.92, 1)))

        btn_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        btn_cancel = Button(text="Cancel", background_normal="", background_down="",
                            background_color=(0.06, 0.13, 0.24, 1), color=(0.85, 0.85, 0.94, 1))
        btn_capture = Button(text="📸  Capture", background_normal="", background_down="",
                             background_color=(0.91, 0.27, 0.38, 1), color=(1, 1, 1, 1),
                             font_size=sp(14), bold=True)
        btn_row.add_widget(btn_cancel)
        btn_row.add_widget(btn_capture)
        content.add_widget(btn_row)

        popup = Popup(title="📷 Capture Image", content=content,
                      size_hint=(0.95, 0.95), background_color=(0.06, 0.08, 0.15, 1),
                      separator_color=(0.91, 0.27, 0.38, 1), title_color=(1, 1, 1, 1))

        def on_capture(inst):
            try:
                tex = cam.texture
                if tex is not None:
                    w, h = tex.size
                    buf = tex.pixels
                    arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                    self._color = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                    self._gray = cv2.cvtColor(self._color, cv2.COLOR_BGR2GRAY)
                    self.image_display.set_texture(cv2_to_texture(self._gray))
                    self.histogram.set_image(self._gray)
                    self.status_label.text = "Image captured ✔"
                    self.status_label.color = (0.4, 0.8, 0.4, 1)
                    cam.play = False
            except Exception as e:
                self._show_error(f"Capture failed:\n{e}")
            popup.dismiss()

        btn_capture.bind(on_press=on_capture)
        btn_cancel.bind(on_press=lambda x: (setattr(cam, 'play', False) if hasattr(cam, 'play') else None,
                                            popup.dismiss()))
        popup.open()

    def _classify_image(self, instance=None):
        if self._gray is None:
            self._show_error("No image loaded. Load or capture an image first.")
            return

        self.status_label.text = "Classifying..."
        self.status_label.color = (0.91, 0.75, 0.27, 1)

        def worker():
            try:
                category, confidence, features = ImageClassifier.classify(self._gray, self._color)
                Clock.schedule_once(lambda dt: self._on_classify_done(category, confidence, features), 0)
            except Exception as e:
                Clock.schedule_once(lambda dt: self._on_classify_error(str(e)), 0)

        threading.Thread(target=worker, daemon=True).start()

    def _on_classify_done(self, category, confidence, features):
        self.status_label.text = "Classification complete ✔"
        self.status_label.color = (0.3, 0.85, 0.3, 1)

        cat_colors = {
            "Fingerprint": (0.29, 0.87, 0.50),
            "Document": (0.38, 0.65, 0.98),
            "Photograph": (0.96, 0.45, 0.71),
            "QR/Barcode": (0.91, 0.47, 0.98),
            "Texture/Pattern": (0.58, 0.77, 0.99),
            "Low Quality": (0.97, 0.44, 0.44),
            "Other": (0.6, 0.6, 0.65),
        }
        self.result_label.text = category
        self.result_label.color = cat_colors.get(category, (1, 1, 1, 1))
        self.result_desc.text = ImageClassifier.CATEGORIES.get(category, "")

        # Update score bars
        all_scores = features.get("all_scores", {})
        for cat, (bar, pct) in self.score_bars.items():
            score = all_scores.get(cat, 0)
            bar.value = int(score * 100)
            pct.text = f"{int(score * 100)}%"

        # Update features
        lines = []
        for k, v in features.items():
            if k == "all_scores":
                continue
            if isinstance(v, float):
                lines.append(f"{k}: {v:.3f}")
            elif isinstance(v, bool):
                lines.append(f"{k}: {'Yes' if v else 'No'}")
            else:
                lines.append(f"{k}: {v}")
        self.features_label.text = "\n".join(lines)

        # Save to history
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        thumb_path = HISTORY_DIR / f"cls_thumb_{ts}.png"
        if self._gray is not None:
            thumb = cv2.resize(self._gray, (100, 100))
            cv2.imwrite(str(thumb_path), thumb)
        history_mgr.add(category, confidence, 0.0, "image_classifier", thumb_path)

    def _on_classify_error(self, error_msg):
        self.status_label.text = "Error ✘"
        self.status_label.color = (0.91, 0.27, 0.27, 1)
        self._show_error(f"Classification failed:\n{error_msg}")

    def _show_error(self, msg):
        popup = Popup(title="Error", content=Label(text=msg, font_size=sp(12)),
                      size_hint=(0.8, 0.4), background_color=(0.06, 0.08, 0.15, 1),
                      separator_color=(0.91, 0.27, 0.38, 1), title_color=(1, 1, 1, 1))
        popup.open()


class HistoryScreen(Screen):
    """Scan history gallery."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "history"

        root = BoxLayout(orientation="vertical")

        # Header
        header = BoxLayout(size_hint_y=None, height=dp(50), padding=[dp(12), dp(6)])
        with header.canvas.before:
            Color(0.05, 0.06, 0.12, 1)
            self._hbg = Rectangle(pos=header.pos, size=header.size)
        header.bind(pos=lambda i, v: setattr(self._hbg, 'pos', v),
                    size=lambda i, v: setattr(self._hbg, 'size', v))
        back_btn = Button(text="← Back", font_size=sp(12), size_hint_x=0.2,
                          background_normal="", background_down="",
                          background_color=(0.15, 0.15, 0.25, 1), color=(0.8, 0.8, 0.86, 1))
        back_btn.bind(on_press=lambda x: setattr(App.get_running_app().sm, 'current', 'home'))
        header.add_widget(back_btn)
        header.add_widget(Label(text="[b]📋 Scan History[/b]", markup=True,
                                font_size=sp(16), color=(0.91, 0.27, 0.38, 1)))

        btn_clear = Button(text="🗑️ Clear", font_size=sp(11), size_hint_x=0.2,
                           background_normal="", background_down="",
                           background_color=(0.91, 0.27, 0.27, 0.3), color=(0.91, 0.6, 0.6, 1))
        btn_clear.bind(on_press=self._clear_history)
        header.add_widget(btn_clear)
        root.add_widget(header)

        # List
        scroll = ScrollView()
        self.list_container = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4),
                                        padding=dp(8))
        self.list_container.bind(minimum_height=self.list_container.setter("height"))
        scroll.add_widget(self.list_container)
        root.add_widget(scroll)
        self.add_widget(root)

    def on_enter(self, *args):
        self._refresh()

    def _refresh(self):
        self.list_container.clear_widgets()
        records = history_mgr.recent(30)
        if not records:
            self.list_container.add_widget(Label(
                text="No scans yet.\nGo to Fingerprint Scanner or Image Classifier to start!",
                font_size=sp(13), color=(0.5, 0.5, 0.55, 1), halign="center"))
            return

        for rec in records:
            card = CardWidget(size_hint_y=None, height=dp(60), spacing=dp(2))
            ts = rec.get("timestamp", "")[:16].replace("T", " ")
            cls = rec.get("classification", "?")
            conf = int(rec.get("confidence", 0) * 100)
            src = "📷" if rec.get("source") == "camera" else ("🎲" if "synthetic" in str(rec.get("source", "")) else "📂")

            row1 = BoxLayout(size_hint_y=None, height=dp(24))
            row1.add_widget(Label(text=f"{src}  {cls}", font_size=sp(13), bold=True,
                                  color=(0.91, 0.85, 0.92, 1), size_hint_x=0.6, halign="left"))
            row1.add_widget(Label(text=f"{conf}%", font_size=sp(12),
                                  color=(0.58, 0.77, 0.99, 1), size_hint_x=0.4, halign="right"))
            card.add_widget(row1)

            row2 = BoxLayout(size_hint_y=None, height=dp(18))
            qual = int(rec.get("quality", 0) * 100)
            row2.add_widget(Label(text=f"🕐 {ts}", font_size=sp(9),
                                  color=(0.5, 0.5, 0.55, 1), size_hint_x=0.6, halign="left"))
            row2.add_widget(Label(text=f"Quality: {qual}%", font_size=sp(9),
                                  color=(0.5, 0.5, 0.55, 1), size_hint_x=0.4, halign="right"))
            card.add_widget(row2)
            self.list_container.add_widget(card)

    def _clear_history(self, instance=None):
        history_mgr.clear()
        self._refresh()


class SettingsScreen(Screen):
    """App settings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "settings"

        root = BoxLayout(orientation="vertical")

        header = BoxLayout(size_hint_y=None, height=dp(50), padding=[dp(12), dp(6)])
        with header.canvas.before:
            Color(0.05, 0.06, 0.12, 1)
            self._hbg = Rectangle(pos=header.pos, size=header.size)
        header.bind(pos=lambda i, v: setattr(self._hbg, 'pos', v),
                    size=lambda i, v: setattr(self._hbg, 'size', v))
        back_btn = Button(text="← Back", font_size=sp(12), size_hint_x=0.2,
                          background_normal="", background_down="",
                          background_color=(0.15, 0.15, 0.25, 1), color=(0.8, 0.8, 0.86, 1))
        back_btn.bind(on_press=lambda x: setattr(App.get_running_app().sm, 'current', 'home'))
        header.add_widget(back_btn)
        header.add_widget(Label(text="[b]⚙️ Settings[/b]", markup=True,
                                font_size=sp(16), color=(0.91, 0.27, 0.38, 1)))
        root.add_widget(header)

        scroll = ScrollView()
        content = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(10), padding=dp(14))
        content.bind(minimum_height=content.setter("height"))

        # About card
        about = CardWidget(size_hint_y=None, height=dp(150))
        about.add_widget(Label(text="ScanClassify", font_size=sp(18), bold=True,
                               color=(0.91, 0.27, 0.38, 1), size_hint_y=None, height=dp(28)))
        about.add_widget(Label(text="Fingerprint & Image Classification App\n\n"
                                    "• Fingerprint: 7 Henry classification categories\n"
                                    "• Image: Auto-detect fingerprint, document, photo, etc.\n"
                                    "• Camera capture for phone fingerprint scans\n"
                                    "• Scan history with export",
                               font_size=sp(11), color=(0.75, 0.75, 0.80, 1),
                               halign="left", valign="top", size_hint_y=None, height=dp(100)))
        content.add_widget(about)

        # Data card
        data = CardWidget(size_hint_y=None, height=dp(90))
        data.add_widget(Label(text="Data", font_size=sp(13), bold=True,
                              color=(0.58, 0.77, 0.99, 1), size_hint_y=None, height=dp(22)))
        n_records = len(history_mgr.records)
        data.add_widget(Label(text=f"Scan history: {n_records} records\n"
                                   f"Storage: {HISTORY_DIR}",
                              font_size=sp(11), color=(0.75, 0.75, 0.80, 1),
                              halign="left", valign="top", size_hint_y=None, height=dp(40)))

        btn_clear = Button(text="🗑️  Clear All History", size_hint_y=None, height=dp(36),
                           background_normal="", background_down="",
                           background_color=(0.91, 0.27, 0.27, 0.4),
                           color=(0.91, 0.7, 0.7, 1), font_size=sp(12))
        btn_clear.bind(on_press=self._clear_history)
        data.add_widget(btn_clear)
        content.add_widget(data)

        # Tips card
        tips = CardWidget(size_hint_y=None, height=dp(160))
        tips.add_widget(Label(text="Tips for Phone Fingerprint Scanning", font_size=sp(13), bold=True,
                              color=(0.58, 0.77, 0.99, 1), size_hint_y=None, height=dp(22)))
        tips.add_widget(Label(text="📱 Place finger gently on the rear camera lens\n"
                                   "💡 Use a bright background light source\n"
                                   "📷 Tap Camera → Capture while finger is on lens\n"
                                   "🔲 The app auto-detects if an image is a fingerprint\n"
                                   "📂 You can also load photos from your gallery\n"
                                   "🎲 Use Generate to test with synthetic patterns",
                              font_size=sp(11), color=(0.75, 0.75, 0.80, 1),
                              halign="left", valign="top", size_hint_y=None, height=dp(120)))
        content.add_widget(tips)

        # Keyboard shortcuts card
        shortcuts = CardWidget(size_hint_y=None, height=dp(70))
        shortcuts.add_widget(Label(text="Keyboard Shortcuts (Desktop)", font_size=sp(13), bold=True,
                                   color=(0.58, 0.77, 0.99, 1), size_hint_y=None, height=dp(22)))
        shortcuts.add_widget(Label(text="Ctrl+O: Open file  |  Ctrl+P: Process\n"
                                        "Ctrl+G: Generate  |  Ctrl+S: Save",
                                   font_size=sp(10), color=(0.5, 0.5, 0.55, 1),
                                   halign="left", valign="top", size_hint_y=None, height=dp(30)))
        content.add_widget(shortcuts)

        scroll.add_widget(content)
        root.add_widget(scroll)
        self.add_widget(root)

    def _clear_history(self, instance=None):
        history_mgr.clear()
        self._show_toast("History cleared")

    def _show_toast(self, msg):
        popup = Popup(title="", content=Label(text=msg, font_size=sp(13)),
                      size_hint=(0.6, 0.15), background_color=(0.1, 0.1, 0.2, 0.9),
                      separator_color=(0, 0, 0, 0))
        popup.open()
        Clock.schedule_once(lambda dt: popup.dismiss(), 1.5)


# ═══════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════

class ScanClassifyApp(App):
    """Main application."""

    def build(self):
        self.title = "ScanClassify"
        self.sm = ScreenManager(transition=SlideTransition(duration=0.25))

        self.sm.add_widget(HomeScreen())
        self.sm.add_widget(FingerprintScreen())
        self.sm.add_widget(ImageClassifyScreen())
        self.sm.add_widget(HistoryScreen())
        self.sm.add_widget(SettingsScreen())

        root = BoxLayout(orientation="vertical")

        # Screen area
        self.screen_area = BoxLayout(orientation="vertical")
        self.screen_area.add_widget(self.sm)
        root.add_widget(self.screen_area)

        # Bottom navigation bar
        nav_bar = BoxLayout(size_hint_y=None, height=dp(56), spacing=dp(0))
        with nav_bar.canvas.before:
            Color(0.04, 0.05, 0.10, 1)
            self._nav_bg = Rectangle(pos=nav_bar.pos, size=nav_bar.size)
        nav_bar.bind(pos=lambda i, v: setattr(self._nav_bg, 'pos', v),
                     size=lambda i, v: setattr(self._nav_bg, 'size', v))

        self.nav_buttons = {}
        nav_items = [
            ("🏠", "Home", "home"),
            ("👆", "Finger", "fingerprint"),
            ("🖼️", "Classify", "classify"),
            ("📋", "History", "history"),
            ("⚙️", "Settings", "settings"),
        ]
        for icon, text, screen in nav_items:
            btn = NavButton(icon=icon, text=text)
            btn.bind(on_press=lambda inst, s=screen: self._navigate(s))
            nav_bar.add_widget(btn)
            self.nav_buttons[screen] = btn

        root.add_widget(nav_bar)

        # Set initial active
        self._update_nav("home")

        # Keyboard shortcuts
        Window.bind(on_keyboard=self._on_keyboard)

        return root

    def _navigate(self, screen_name):
        self.sm.current = screen_name
        self._update_nav(screen_name)

    def _update_nav(self, active_screen):
        for name, btn in self.nav_buttons.items():
            btn.active = (name == active_screen)

    def _on_keyboard(self, instance, key, scancode, codepoint, modifier):
        ctrl = "ctrl" in modifier
        if ctrl and codepoint == "o":
            # Open file on current relevant screen
            current = self.sm.current
            if current == "fingerprint":
                self.sm.get_screen("fingerprint")._show_file_chooser()
            elif current == "classify":
                self.sm.get_screen("classify")._show_file_chooser()
        elif ctrl and codepoint == "p":
            if self.sm.current == "fingerprint":
                self.sm.get_screen("fingerprint")._process_image()
        elif ctrl and codepoint == "g":
            if self.sm.current == "fingerprint":
                self.sm.get_screen("fingerprint")._generate_pattern()
        elif ctrl and codepoint == "s":
            if self.sm.current == "fingerprint":
                self.sm.get_screen("fingerprint")._save_enhanced()


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ScanClassifyApp().run()