# AirfoilExtractor

# -*- coding: utf-8 -*-
"""
Airfoil Parameter Extractor (Desktop GUI + Snipping Tool Overlay)
-----------------------------------------------------------------
cv2(opencv-python) 의존성을 완전히 제거한 경량 버전입니다.
실행 중인 파이썬 환경에 필수 패키지가 없으면 자동으로 설치를 시도합니다.

단일 .exe 빌드 명령어 (터미널):
    python -m PyInstaller --onefile --noconsole --name AirfoilExtractor airfoil_extractor_app.py
"""

import sys
import os
import csv
import subprocess
from math import comb
import ctypes

# 현재 실행 중인 Python 인터프리터에 필수 패키지가 없으면 자동 설치
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "mss": "mss",
    "PySide6": "PySide6",
    "matplotlib": "matplotlib",
    "PyInstaller": "pyinstaller",
}

if not getattr(sys, "frozen", False):
    for mod_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(mod_name)
        except ImportError:
            print(f"[자동 설치] '{pip_name}' 패키지를 현재 Python 환경에 설치합니다...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

import numpy as np
import mss
from scipy.interpolate import PchipInterpolator
from scipy.signal import medfilt
from scipy.optimize import least_squares

from PySide6.QtCore import Qt, QRect, QPoint, QPointF, QTimer, Signal
from PySide6.QtGui import (
    QPixmap,
    QImage,
    QPainter,
    QColor,
    QPen,
    QBrush,
    QPolygonF,
    QGuiApplication,
    QFont,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QSlider,
    QFileDialog,
    QFrame,
    QGridLayout,
    QMessageBox,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Windows 고해상도(DPI) 배율 보정
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

# 한글 폰트 설정 (Windows 맑은 고딕 우선)
matplotlib.rcParams["axes.unicode_minus"] = False
if sys.platform == "win32":
    matplotlib.rcParams["font.family"] = "Malgun Gothic"


def qimage_to_rgb_array(qimg: QImage) -> np.ndarray:
    """QImage를 (H, W, 3) uint8 RGB numpy 배열로 변환 (cv2 불필요)"""
    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    bpl = qimg.bytesPerLine()
    ptr = qimg.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))[:, : w * 3]
    return arr.reshape((h, w, 3)).copy()


# =====================================================================
# 1. 캡처 도구(Snipping Tool) 전체 화면 드래그 오버레이 창
# =====================================================================
class SnippingOverlay(QWidget):
    cropped_signal = Signal(np.ndarray)
    canceled_signal = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setCursor(Qt.CursorShape.CrossCursor)

        self.full_rgb = None
        self.bg_pixmap = None
        self.dpr = 1.0
        self.start_pos = None
        self.end_pos = None
        self.is_dragging = False

    def start_capture(self):
        """현재 모니터를 Windows DPI 배율(125%/150% 등) 확대 없이 원본 크기로 캡처"""
        screen = QGuiApplication.primaryScreen()
        # Qt 내장 grabWindow(0)는 현재 화면의 devicePixelRatio 메타데이터를 자동 보존함
        raw_pixmap = screen.grabWindow(0)
        self.dpr = float(raw_pixmap.devicePixelRatio())
        qimg = raw_pixmap.toImage()
        self.full_rgb = qimage_to_rgb_array(qimg)

        self.bg_pixmap = raw_pixmap
        # 논리적 화면 좌표계(geometry)와 1:1 일치시켜 화면 확대 현상 방지
        self.setGeometry(screen.geometry())
        self.start_pos = None
        self.end_pos = None
        self.is_dragging = False
        self.show()
        self.activateWindow()
        self.raise_()

    def paintEvent(self, event):
        if self.bg_pixmap is None:
            return
        painter = QPainter(self)
        # bg_pixmap에 devicePixelRatio가 설정되어 있어 (0, 0)에 그리면 원래 화면 크기로 정확히 렌더링됨
        painter.drawPixmap(0, 0, self.bg_pixmap)
        painter.fillRect(self.rect(), QColor(15, 23, 42, 145))

        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Malgun Gothic", 11, QFont.Weight.Bold))
        painter.drawText(
            24,
            36,
            "마우스를 드래그하여 분석할 에어포일 영역을 선택하세요 (취소: ESC)",
        )

        if self.start_pos and self.end_pos:
            rect = QRect(self.start_pos, self.end_pos).normalized()
            if rect.width() > 2 and rect.height() > 2:
                # 고해상도(DPI) 원본 픽셀 좌표계에서 해당 영역만 추출해 그리기
                src_rect = QRect(
                    int(round(rect.x() * self.dpr)),
                    int(round(rect.y() * self.dpr)),
                    int(round(rect.width() * self.dpr)),
                    int(round(rect.height() * self.dpr)),
                )
                painter.drawPixmap(rect, self.bg_pixmap, src_rect)
                pen = QPen(QColor(56, 189, 248), 2, Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                painter.drawRect(rect)

                pw = int(round(rect.width() * self.dpr))
                ph = int(round(rect.height() * self.dpr))
                size_txt = f"{pw} × {ph} px"
                painter.fillRect(
                    rect.left(),
                    max(0, rect.top() - 26),
                    135,
                    22,
                    QColor(15, 23, 42, 210),
                )
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
                painter.drawText(
                    rect.left() + 6, max(15, rect.top() - 10), size_txt
                )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.start_pos = event.position().toPoint()
            self.end_pos = self.start_pos
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.end_pos = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.end_pos = event.position().toPoint()
            rect = QRect(self.start_pos, self.end_pos).normalized()
            self.hide()

            if rect.width() < 12 or rect.height() < 8:
                self.canceled_signal.emit()
                return

            # 논리 좌표를 물리 픽셀 좌표(dpr 반영)로 변환해 정확히 크롭
            x1 = max(0, int(round(rect.left() * self.dpr)))
            y1 = max(0, int(round(rect.top() * self.dpr)))
            x2 = min(self.full_rgb.shape[1], int(round(rect.right() * self.dpr)))
            y2 = min(self.full_rgb.shape[0], int(round(rect.bottom() * self.dpr)))
            cropped = self.full_rgb[y1:y2, x1:x2].copy()
            self.cropped_signal.emit(cropped)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            self.canceled_signal.emit()


# =====================================================================
# 2. 에어포일 외곽선 추출 및 정규화 파라미터 계산 엔진
# =====================================================================
def fit_circle_kasa(pts):
    """앞전(LE) 점군에 대한 최소자승 원 피팅"""
    if len(pts) < 4:
        return 0.02, 0.0, 0.02
    x = pts[:, 0]
    y = pts[:, 1]
    z = x * x + y * y
    A = np.column_stack([x, y, np.ones_like(x)])
    sol, _, _, _ = np.linalg.lstsq(A, -z, rcond=None)
    xc = -0.5 * sol[0]
    yc = -0.5 * sol[1]
    r = np.sqrt(max(1e-6, xc * xc + yc * yc - sol[2]))
    return xc, yc, r


def cst_matrix(x_arr, order=4):
    """CST(Class-Shape Transformation) 기저 행렬 생성 (N1=0.5, N2=1.0)"""
    x_c = np.clip(x_arr, 0.0, 1.0)
    C = np.sqrt(x_c) * (1.0 - x_c)
    B = np.zeros((len(x_c), order + 1))
    for i in range(order + 1):
        B[:, i] = C * comb(order, i) * (x_c**i) * ((1.0 - x_c) ** (order - i))
    return B


def fit_cst_robust(x_pts, y_pts, x_eval, order=4):
    """
    Huber 손실 기반 로버스트(Robust) CST 피팅:
    흐릿한 선, 그리드/캠버선 간섭으로 튀는 지그재그 이상치(Outlier)를 무시하고
    매끄러운 에어포일 형상 곡선과 정상 인라이어(Inlier) 포인트만 추출합니다.
    """
    B_pts = cst_matrix(x_pts, order=order)
    # 초기치: 일반 최소자승법
    w0, _, _, _ = np.linalg.lstsq(B_pts, y_pts, rcond=None)

    # Huber loss 최적화로 이상치(지그재그 점) 영향 배제 + 2차 미분(곡률 급변) 정칙화
    def residuals(w):
        fit_err = B_pts @ w - y_pts
        reg_err = 0.015 * np.diff(w, n=2)
        return np.concatenate([fit_err, reg_err])

    res = least_squares(residuals, w0, loss="huber", f_scale=0.008)
    w_opt = res.x

    y_pred_pts = B_pts @ w_opt
    abs_err = np.abs(y_pts - y_pred_pts)
    mad = max(1e-4, float(np.median(abs_err)))
    inliers = abs_err <= max(0.012, 2.5 * mad)

    # 정상 포인트(Inlier)만으로 한 번 더 정밀 피팅
    if np.sum(inliers) >= order + 2:
        res2 = least_squares(
            lambda w: np.concatenate([B_pts[inliers] @ w - y_pts[inliers], 0.01 * np.diff(w, n=2)]),
            w_opt,
            loss="soft_l1",
            f_scale=0.005,
        )
        w_opt = res2.x

    B_eval = cst_matrix(x_eval, order=order)
    y_eval = B_eval @ w_opt
    return w_opt, y_eval, inliers


def clean_raw_curve(pts):
    """추출된 픽셀 점군에서 급격히 튀는 지그재그 스파이크를 1차 제거"""
    if len(pts) < 7:
        return pts
    xs = pts[:, 0]
    ys = pts[:, 1]
    k = min(11, len(ys) if len(ys) % 2 == 1 else len(ys) - 1)
    if k >= 3:
        y_med = medfilt(ys, kernel_size=k)
        diff = np.abs(ys - y_med)
        thresh_dev = max(2.5, 2.5 * float(np.median(diff)))
        keep = diff <= thresh_dev
        if np.sum(keep) >= 6:
            xs, ys = xs[keep], ys[keep]
    return np.column_stack([xs, ys])


def analyze_airfoil_image(rgb_img, mode="auto", align_mode="chord", thresh=40.0):
    """
    RGB 캡처 이미지에서 에어포일 상·하면을 추출하고,
    불분명하거나 지그재그로 튀는 구간은 로버스트 CST 에어포일 모델로 내삽·보정합니다.
    """
    h, w, _ = rgb_img.shape
    R = rgb_img[:, :, 0].astype(np.float64)
    G = rgb_img[:, :, 1].astype(np.float64)
    B = rgb_img[:, :, 2].astype(np.float64)

    red_diff = R - np.maximum(G, B)
    if mode == "auto":
        red_pixels = np.sum(red_diff > 45.0)
        active_mode = "red" if red_pixels > max(30, w * 0.25) else "dark"
    else:
        active_mode = mode

    if active_mode == "red":
        score = np.maximum(0.0, red_diff)
    else:
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        border_vals = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        bg_gray = float(np.median(border_vals))
        score = np.maximum(0.0, bg_gray - gray)

        # 내부가 흰색(>235)으로 채워진 에어포일(예: AxCent 기본 뷰)이면 흰색 내부 경계를 보조 활용
        white_mask = (gray > max(235.0, bg_gray + 12.0)).astype(np.float64)
        if np.sum(white_mask) > (h * w * 0.03):
            edge_v = np.abs(np.diff(white_mask, axis=0, prepend=0))
            edge_h = np.abs(np.diff(white_mask, axis=1, prepend=0))
            white_edge = np.clip(edge_v + edge_h, 0.0, 1.0) * 80.0
            score = np.maximum(score, white_edge)

    col_max = score.max(axis=0)
    valid_cols = np.where(col_max > thresh)[0]
    if len(valid_cols) < 15:
        raise ValueError("에어포일 윤곽선을 찾지 못했습니다. 캡처 영역이나 민감도를 확인해 주세요.")

    # 좌우 고립된 1~2픽셀 노이즈를 무시하고 가장 긴 연속 유효 구간(에어포일 본체) 선택
    gaps = np.where(np.diff(valid_cols) > 5)[0]
    splits = np.split(valid_cols, gaps + 1)
    main_span = max(splits, key=len)
    x_min, x_max = int(main_span[0]), int(main_span[-1])

    raw_upper = []
    raw_lower = []
    for x in range(x_min, x_max + 1):
        col = score[:, x]
        ys_valid = np.where(col > thresh)[0]
        if len(ys_valid) == 0:
            continue

        # 가장 위쪽 외곽선 피크(Upper)와 가장 아래쪽 외곽선 피크(Lower)를 찾아 내부 캠버선/그리드 간섭 방지
        y_top_seed = int(ys_valid[0])
        y0_u, y1_u = max(0, y_top_seed - 2), min(h, y_top_seed + 4)
        idx_u = y0_u + int(np.argmax(col[y0_u:y1_u]))
        y0, y1 = max(0, idx_u - 2), min(h, idx_u + 3)
        ws_u = np.maximum(0.0, col[y0:y1] - thresh * 0.4)
        if np.sum(ws_u) > 0:
            raw_upper.append([float(x), float(np.sum(np.arange(y0, y1) * ws_u) / np.sum(ws_u))])

        y_bot_seed = int(ys_valid[-1])
        y0_l, y1_l = max(0, y_bot_seed - 3), min(h, y_bot_seed + 3)
        idx_l = y0_l + int(np.argmax(col[y0_l:y1_l]))
        y0, y1 = max(0, idx_l - 2), min(h, idx_l + 3)
        ws_l = np.maximum(0.0, col[y0:y1] - thresh * 0.4)
        if np.sum(ws_l) > 0:
            raw_lower.append([float(x), float(np.sum(np.arange(y0, y1) * ws_l) / np.sum(ws_l))])

    if len(raw_upper) < 10 or len(raw_lower) < 10:
        raise ValueError("상·하면 외곽선이 충분히 검출되지 않았습니다.")

    # 1차 지그재그 스파이크 필터링
    raw_upper = clean_raw_curve(np.array(raw_upper))
    raw_lower = clean_raw_curve(np.array(raw_lower))

    # 앞전(LE)과 뒷전(TE) 끝점: 끝단 3% 구간의 상·하면 중점(Camber) 추세선으로부터 정확히 외삽
    span_w = max(1.0, float(x_max - x_min))
    le_band_u = raw_upper[raw_upper[:, 0] <= x_min + 0.03 * span_w]
    le_band_l = raw_lower[raw_lower[:, 0] <= x_min + 0.03 * span_w]
    te_band_u = raw_upper[raw_upper[:, 0] >= x_max - 0.03 * span_w]
    te_band_l = raw_lower[raw_lower[:, 0] >= x_max - 0.03 * span_w]

    y_le = 0.5 * (float(np.mean(le_band_u[:, 1])) + float(np.mean(le_band_l[:, 1]))) if (len(le_band_u) and len(le_band_l)) else 0.5 * (raw_upper[0, 1] + raw_lower[0, 1])
    y_te = 0.5 * (float(np.mean(te_band_u[:, 1])) + float(np.mean(te_band_l[:, 1]))) if (len(te_band_u) and len(te_band_l)) else 0.5 * (raw_upper[-1, 1] + raw_lower[-1, 1])

    # 최장 현(Chord) 기준 LE 정밀 보정: TE에서 가장 먼 외곽선 점을 LE로 설정
    all_raw = np.vstack([raw_upper, raw_lower])
    te_pt = np.array([float(x_max), float(y_te)])
    dists_te = np.hypot(all_raw[:, 0] - te_pt[0], all_raw[:, 1] - te_pt[1])
    le_pt = all_raw[int(np.argmax(dists_te))].copy()

    dx_c = te_pt[0] - le_pt[0]
    dy_c = -(te_pt[1] - le_pt[1])
    chord_px = float(np.hypot(dx_c, dy_c))
    tilt_deg = float(np.degrees(np.arctan2(dy_c, dx_c)))
    tilt_rad = np.radians(tilt_deg) if align_mode == "chord" else 0.0

    cos_a, sin_a = np.cos(-tilt_rad), np.sin(-tilt_rad)

    def to_norm_pts(pts):
        mx = pts[:, 0] - le_pt[0]
        my = -(pts[:, 1] - le_pt[1])
        rx = (mx * cos_a - my * sin_a) / chord_px
        ry = (mx * sin_a + my * cos_a) / chord_px
        mask = (rx >= 0.002) & (rx <= 0.998)
        rx, ry = rx[mask], ry[mask]
        order = np.argsort(rx)
        return rx[order], ry[order]

    xu_n, yu_n = to_norm_pts(raw_upper)
    xl_n, yl_n = to_norm_pts(raw_lower)

    # 코사인 간격 정규화 격자
    theta = np.linspace(0.0, np.pi, 301)
    x_grid = 0.5 * (1.0 - np.cos(theta))

    # 2차: 로버스트 CST 에어포일 기하 모델 피팅 (지그재그 완벽 제거 및 매끄러운 에어포일 형상 보장)
    w_u, yu_cst, inliers_u = fit_cst_robust(xu_n, yu_n, x_grid, order=4)
    w_l, yl_cst, inliers_l = fit_cst_robust(xl_n, yl_n, x_grid, order=4)

    # 상면이 하면보다 항상 위에 오도록 최소 두께 보장
    yu_g = yu_cst.copy()
    yl_g = np.minimum(yl_cst, yu_g - 0.001 * np.sin(theta))
    yu_g[0], yu_g[-1] = 0.0, 0.0
    yl_g[0], yl_g[-1] = 0.0, 0.0

    # 원본 이미지 위에 오버레이할 매끄러운 에어포일 외곽선 역변환 (시각화용)
    def norm_to_image(xn, yn):
        mx = (xn * cos_a + yn * sin_a) * chord_px
        my = (-xn * sin_a + yn * cos_a) * chord_px
        ix = le_pt[0] + mx
        iy = le_pt[1] - my
        return np.column_stack([ix, iy])

    smooth_upper_img = norm_to_image(x_grid, yu_g)
    smooth_lower_img = norm_to_image(x_grid, yl_g)

    yc_g = 0.5 * (yu_g + yl_g)
    t_g = yu_g - yl_g

    idx_mc = int(np.argmax(yc_g))
    max_camber = float(yc_g[idx_mc])
    max_camber_pos = float(x_grid[idx_mc])

    idx_mt = int(np.argmax(t_g))
    max_thick = float(t_g[idx_mt])
    max_thick_pos = float(x_grid[idx_mt])

    # CST 수식으로부터 해석적 LE 반경 및 매끄러운 원 피팅 계산
    le_mask = (x_grid >= 0.001) & (x_grid <= 0.035)
    le_pts = np.vstack(
        [np.column_stack([x_grid[le_mask], yu_g[le_mask]]), np.column_stack([x_grid[le_mask], yl_g[le_mask]])]
    )
    xc_le, yc_le, r_le = fit_circle_kasa(le_pts)

    def val_at(arr, tx):
        return float(arr[int(np.argmin(np.abs(x_grid - tx)))])

    chi_in = float(np.degrees(np.arctan((val_at(yc_g, 0.05) - val_at(yc_g, 0.005)) / 0.045)))
    chi_out = float(np.degrees(np.arctan(-(val_at(yc_g, 0.995) - val_at(yc_g, 0.95)) / 0.045)))
    camber_angle = chi_in + chi_out

    mask_te = (x_grid >= 0.90) & (x_grid <= 0.99)
    m_u = np.polyfit(x_grid[mask_te], yu_g[mask_te], 1)[0]
    m_l = np.polyfit(x_grid[mask_te], yl_g[mask_te], 1)[0]
    te_wedge = float(np.degrees(np.arctan(m_l) - np.arctan(m_u)))

    # 인라이어(유효 포인트) 기준 피팅 오차 계산
    u_err = (cst_matrix(xu_n[inliers_u], 4) @ w_u - yu_n[inliers_u]) if np.any(inliers_u) else np.zeros(1)
    l_err = (cst_matrix(xl_n[inliers_l], 4) @ w_l - yl_n[inliers_l]) if np.any(inliers_l) else np.zeros(1)
    cst_rmse = float(np.sqrt(0.5 * (np.mean(u_err**2) + np.mean(l_err**2))))

    return {
        "active_mode": active_mode,
        "chord_px": chord_px,
        "tilt_deg": tilt_deg,
        "max_camber": max_camber,
        "max_camber_pos": max_camber_pos,
        "max_thick": max_thick,
        "max_thick_pos": max_thick_pos,
        "le_radius": float(r_le),
        "le_center": (float(xc_le), float(yc_le)),
        "chi_in": chi_in,
        "chi_out": chi_out,
        "camber_angle": camber_angle,
        "te_wedge": te_wedge,
        "w_u": w_u,
        "w_l": w_l,
        "cst_rmse": cst_rmse,
        "x_grid": x_grid,
        "yu": yu_g,
        "yl": yl_g,
        "yc": yc_g,
        "thick": t_g,
        "raw_upper": smooth_upper_img,
        "raw_lower": smooth_lower_img,
        "pts_upper_inlier": raw_upper[inliers_u] if len(inliers_u) == len(raw_upper) else raw_upper,
        "pts_lower_inlier": raw_lower[inliers_l] if len(inliers_l) == len(raw_lower) else raw_lower,
        "le_pt": le_pt,
        "te_pt": te_pt,
    }


# =====================================================================
# 3. 메인 데스크톱 GUI 창
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Airfoil Parameter Extractor — 캡처 기반 에어포일 파라미터 추출기")
        self.resize(1240, 740)
        self.setStyleSheet(
            """
            QMainWindow { background-color: #f8fafc; }
            QFrame#Card {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }
            QPushButton {
                padding: 8px 14px;
                border-radius: 7px;
                font-weight: 600;
                font-size: 12px;
            }
            """
        )

        self.current_rgb = None
        self.last_result = None
        self.snipper = SnippingOverlay()
        self.snipper.cropped_signal.connect(self.on_snip_completed)
        self.snipper.canceled_signal.connect(self.on_snip_canceled)

        self._init_ui()
        self.load_demo_sample()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # --- 상단 툴바 영역 ---
        top_card = QFrame()
        top_card.setObjectName("Card")
        top_layout = QHBoxLayout(top_card)
        top_layout.setContentsMargins(14, 10, 14, 10)
        top_layout.setSpacing(10)

        self.btn_snip = QPushButton("✂ 화면 캡처 (영역 드래그)")
        self.btn_snip.setStyleSheet("background-color: #059669; color: white;")
        self.btn_snip.clicked.connect(self.start_snip_mode)

        self.btn_open = QPushButton("📂 이미지 파일 열기")
        self.btn_open.setStyleSheet("background-color: #2563eb; color: white;")
        self.btn_open.clicked.connect(self.open_image_file)

        self.btn_paste = QPushButton("📋 클립보드 붙여넣기 (Ctrl+V)")
        self.btn_paste.setStyleSheet("background-color: #e2e8f0; color: #1e293b;")
        self.btn_paste.clicked.connect(self.paste_from_clipboard)

        self.btn_demo = QPushButton("샘플 불러오기")
        self.btn_demo.setStyleSheet("background-color: #f1f5f9; color: #334155;")
        self.btn_demo.clicked.connect(self.load_demo_sample)

        top_layout.addWidget(self.btn_snip)
        top_layout.addWidget(self.btn_open)
        top_layout.addWidget(self.btn_paste)
        top_layout.addWidget(self.btn_demo)
        top_layout.addStretch()

        top_layout.addWidget(QLabel("추출 모드:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["자동 감지 (Auto)", "어두운 외곽선 (AxCent)", "붉은색 선 (그리드 제외)"])
        self.combo_mode.currentIndexChanged.connect(self.reprocess_current)
        top_layout.addWidget(self.combo_mode)

        top_layout.addWidget(QLabel("정렬:"))
        self.combo_align = QComboBox()
        self.combo_align.addItems(["자동 코드선 정렬 (LE→TE)", "수평 기준 유지"])
        self.combo_align.currentIndexChanged.connect(self.reprocess_current)
        top_layout.addWidget(self.combo_align)

        top_layout.addWidget(QLabel("민감도:"))
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(15, 140)
        self.slider_thresh.setValue(40)
        self.slider_thresh.setFixedWidth(100)
        self.slider_thresh.valueChanged.connect(self.on_thresh_changed)
        top_layout.addWidget(self.slider_thresh)

        self.lbl_thresh = QLabel("40")
        self.lbl_thresh.setFixedWidth(26)
        top_layout.addWidget(self.lbl_thresh)

        self.btn_csv = QPushButton("💾 CSV 저장")
        self.btn_csv.setStyleSheet("background-color: #0f172a; color: white;")
        self.btn_csv.clicked.connect(self.save_csv)
        top_layout.addWidget(self.btn_csv)

        main_layout.addWidget(top_card)

        # --- 중단 요약 파라미터 카드 그리드 ---
        summary_grid = QGridLayout()
        summary_grid.setSpacing(10)
        self.metric_labels = {}
        metrics_info = [
            ("max_camber", "Max Camber (%)"),
            ("max_camber_pos", "Max Camber Position (%)"),
            ("max_thick", "Max Thickness (%)"),
            ("le_radius", "LE Radius (%c)"),
            ("camber_angle", "Total Camber Angle (입/출구)"),
            ("te_wedge", "TE Wedge / 이미지 기울기"),
        ]
        for idx, (key, title) in enumerate(metrics_info):
            card = QFrame()
            card.setObjectName("Card")
            vbox = QVBoxLayout(card)
            vbox.setContentsMargins(12, 10, 12, 10)
            lbl_title = QLabel(title)
            lbl_title.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")
            lbl_val = QLabel("-")
            lbl_val.setStyleSheet("color: #0f172a; font-size: 18px; font-weight: 700;")
            lbl_sub = QLabel("-")
            lbl_sub.setStyleSheet("color: #059669; font-size: 11px;")
            vbox.addWidget(lbl_title)
            vbox.addWidget(lbl_val)
            vbox.addWidget(lbl_sub)
            summary_grid.addWidget(card, 0, idx)
            self.metric_labels[key] = (lbl_val, lbl_sub)

        main_layout.addLayout(summary_grid)

        # --- 하단 나란히 비교 뷰 ---
        plot_card = QFrame()
        plot_card.setObjectName("Card")
        plot_layout = QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(8, 8, 8, 8)

        self.fig = Figure(figsize=(12, 5.2), dpi=100, facecolor="#ffffff")
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)

        self.lbl_cst = QLabel("4차 CST 계수: -")
        self.lbl_cst.setStyleSheet(
            "color: #334155; font-family: Consolas, monospace; font-size: 11px; padding: 4px 8px;"
        )
        plot_layout.addWidget(self.lbl_cst)

        main_layout.addWidget(plot_card, stretch=1)

    def start_snip_mode(self):
        self.showMinimized()
        QTimer.singleShot(280, self.snipper.start_capture)

    def on_snip_completed(self, cropped_rgb):
        self.showNormal()
        self.activateWindow()
        self.current_rgb = cropped_rgb
        self.reprocess_current()

    def on_snip_canceled(self):
        self.showNormal()
        self.activateWindow()

    def open_image_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "에어포일 이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        qimg = QImage(path)
        if not qimg.isNull():
            self.current_rgb = qimage_to_rgb_array(qimg)
            self.reprocess_current()

    def paste_from_clipboard(self):
        clipboard = QGuiApplication.clipboard()
        qimg = clipboard.image()
        if qimg.isNull():
            QMessageBox.information(self, "알림", "클립보드에 이미지가 없습니다.")
            return
        self.current_rgb = qimage_to_rgb_array(qimg)
        self.reprocess_current()

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_V:
            self.paste_from_clipboard()

    def on_thresh_changed(self, val):
        self.lbl_thresh.setText(str(val))
        self.reprocess_current()

    def load_demo_sample(self):
        """QPainter로 샘플 에어포일 이미지 생성 (cv2 불필요)"""
        w, h = 640, 180
        qimg = QImage(w, h, QImage.Format.Format_RGB888)
        qimg.fill(QColor(222, 226, 232))

        m, p, t = 0.055, 0.36, 0.165
        c, x0, y0 = 510.0, 65.0, 112.0
        betas = np.linspace(0, np.pi, 200)
        xs = 0.5 * (1.0 - np.cos(betas))
        yt = 5 * t * (0.2969 * np.sqrt(xs) - 0.1260 * xs - 0.3516 * xs**2 + 0.2843 * xs**3 - 0.1015 * xs**4)
        yc = np.where(
            xs < p,
            (m / p**2) * (2 * p * xs - xs**2),
            (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xs - xs**2),
        )
        dyc = np.where(
            xs < p,
            (2 * m / p**2) * (p - xs),
            (2 * m / (1 - p) ** 2) * (p - xs),
        )
        th = np.arctan(dyc)
        ux = x0 + (xs - yt * np.sin(th)) * c
        uy = y0 - (yc + yt * np.cos(th)) * c
        lx = x0 + (xs + yt * np.sin(th)) * c
        ly = y0 - (yc - yt * np.cos(th)) * c

        pts = np.vstack([np.column_stack([ux, uy]), np.column_stack([lx[::-1], ly[::-1]])])
        poly = QPolygonF([QPointF(float(px), float(py)) for px, py in pts])

        painter = QPainter(qimg)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(45, 50, 60), 2.2))
        painter.drawPolygon(poly)
        painter.end()

        self.current_rgb = qimage_to_rgb_array(qimg)
        self.reprocess_current()

    def reprocess_current(self):
        if self.current_rgb is None:
            return
        mode_map = {0: "auto", 1: "dark", 2: "red"}
        align_map = {0: "chord", 1: "horizontal"}
        mode = mode_map.get(self.combo_mode.currentIndex(), "auto")
        align = align_map.get(self.combo_align.currentIndex(), "chord")
        thresh = float(self.slider_thresh.value())

        try:
            res = analyze_airfoil_image(self.current_rgb, mode=mode, align_mode=align, thresh=thresh)
            self.last_result = res
            self._update_summary_cards(res)
            self._draw_side_by_side(res)
        except Exception as e:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.imshow(self.current_rgb)
            ax.set_title(str(e), color="crimson", fontsize=11)
            ax.axis("off")
            self.canvas.draw()

    def _update_summary_cards(self, r):
        self.metric_labels["max_camber"][0].setText(f"{r['max_camber'] * 100:.2f}%")
        self.metric_labels["max_camber"][1].setText(f"모드: {r['active_mode'].upper()}")

        self.metric_labels["max_camber_pos"][0].setText(f"{r['max_camber_pos'] * 100:.1f}%")
        self.metric_labels["max_camber_pos"][1].setText("전연(LE) 기준 코드 비율")

        self.metric_labels["max_thick"][0].setText(f"{r['max_thick'] * 100:.2f}%")
        self.metric_labels["max_thick"][1].setText(f"위치: {r['max_thick_pos'] * 100:.1f}% chord")

        self.metric_labels["le_radius"][0].setText(f"{r['le_radius'] * 100:.2f}%c")
        self.metric_labels["le_radius"][1].setText(f"코드 길이: {r['chord_px']:.1f} px")

        self.metric_labels["camber_angle"][0].setText(f"{r['camber_angle']:.1f}°")
        self.metric_labels["camber_angle"][1].setText(f"입구 {r['chi_in']:.1f}° / 출구 {r['chi_out']:.1f}°")

        self.metric_labels["te_wedge"][0].setText(f"{r['te_wedge']:.1f}°")
        self.metric_labels["te_wedge"][1].setText(f"기울기: {r['tilt_deg']:.2f}°")

        wu_str = ", ".join(f"{v:.4f}" for v in r["w_u"])
        wl_str = ", ".join(f"{v:.4f}" for v in r["w_l"])
        self.lbl_cst.setText(
            f"4차 CST Upper: [{wu_str}]   |   Lower: [{wl_str}]   |   RMSE: {r['cst_rmse']*100:.3f}%c"
        )

    def _draw_side_by_side(self, r):
        self.fig.clear()
        ax_orig, ax_norm = self.fig.subplots(1, 2)
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.14, wspace=0.18)

        ax_orig.imshow(self.current_rgb)
        # 유효하게 채택된 포인트(연한 점)와 보정된 에어포일 곡선(실선) 표시
        u_in = r.get("pts_upper_inlier", r["raw_upper"])
        l_in = r.get("pts_lower_inlier", r["raw_lower"])
        ax_orig.scatter(u_in[::4, 0], u_in[::4, 1], c="salmon", s=8, alpha=0.6)
        ax_orig.scatter(l_in[::4, 0], l_in[::4, 1], c="skyblue", s=8, alpha=0.6)
        ax_orig.plot(
            r["raw_upper"][:, 0], r["raw_upper"][:, 1], "r-", lw=1.8, label="보정 상면 (Upper)"
        )
        ax_orig.plot(
            r["raw_lower"][:, 0], r["raw_lower"][:, 1], "b-", lw=1.8, label="보정 하면 (Lower)"
        )
        ax_orig.plot(
            [r["le_pt"][0], r["te_pt"][0]],
            [r["le_pt"][1], r["te_pt"][1]],
            "g--",
            lw=1.3,
            label=f"코드선 ({r['chord_px']:.1f}px)",
        )
        ax_orig.scatter(
            [r["le_pt"][0], r["te_pt"][0]],
            [r["le_pt"][1], r["te_pt"][1]],
            c=["lime", "cyan"],
            s=40,
            edgecolors="black",
            zorder=5,
        )
        ax_orig.set_title("1. 캡처 원본 및 에어포일 선 검출", fontsize=11, fontweight="bold")
        ax_orig.legend(loc="upper right", fontsize=8.5)
        ax_orig.axis("off")

        xg = r["x_grid"] * 100.0
        yu = r["yu"] * 100.0
        yl = r["yl"] * 100.0
        yc = r["yc"] * 100.0

        ax_norm.plot(xg, yu, "r-", lw=2.0, label="상면 (Upper)")
        ax_norm.plot(xg, yl, "b-", lw=2.0, label="하면 (Lower)")
        ax_norm.plot(xg, yc, "g--", lw=1.6, label="평균 캠버선 (Camber)")
        ax_norm.axhline(0, color="black", linestyle=":", lw=1.0)

        idx_mt = int(np.argmax(r["thick"]))
        mt_x = r["max_thick_pos"] * 100.0
        ax_norm.plot(
            [mt_x, mt_x],
            [yl[idx_mt], yu[idx_mt]],
            "m-",
            lw=2.2,
            label=f"Max Thickness: {r['max_thick']*100:.2f}% (@{mt_x:.1f}%)",
        )
        mc_x = r["max_camber_pos"] * 100.0
        mc_y = r["max_camber"] * 100.0
        ax_norm.scatter(
            [mc_x],
            [mc_y],
            color="green",
            s=45,
            zorder=5,
            label=f"Max Camber: {mc_y:.2f}% (@{mc_x:.1f}%)",
        )

        ax_norm.set_aspect("equal", adjustable="datalim")
        ax_norm.set_xlim(-3, 103)
        ax_norm.grid(True, alpha=0.3)
        ax_norm.set_xlabel("Normalized Chord (%c)")
        ax_norm.set_ylabel("y/c (%)")
        ax_norm.set_title("2. 정규화 에어포일 형상 및 파라미터", fontsize=11, fontweight="bold")
        ax_norm.legend(loc="upper right", fontsize=8.0)

        self.canvas.draw()

    def save_csv(self):
        if self.last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "파라미터 및 정규화 좌표 CSV 저장", "airfoil_parameters.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        r = self.last_result
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Value", "Unit"])
            writer.writerow(["Max Camber", f"{r['max_camber']*100:.4f}", "%"])
            writer.writerow(["Max Camber Position", f"{r['max_camber_pos']*100:.2f}", "%"])
            writer.writerow(["Max Thickness", f"{r['max_thick']*100:.4f}", "%"])
            writer.writerow(["Max Thickness Position", f"{r['max_thick_pos']*100:.2f}", "%"])
            writer.writerow(["LE Radius", f"{r['le_radius']*100:.4f}", "%c"])
            writer.writerow(["Total Camber Angle", f"{r['camber_angle']:.2f}", "deg"])
            writer.writerow(["TE Wedge Angle", f"{r['te_wedge']:.2f}", "deg"])
            writer.writerow([])
            writer.writerow(["x/c", "y_upper/c", "y_lower/c", "y_camber/c", "thickness/c"])
            for i in range(len(r["x_grid"])):
                writer.writerow(
                    [
                        f"{r['x_grid'][i]:.6f}",
                        f"{r['yu'][i]:.6f}",
                        f"{r['yl'][i]:.6f}",
                        f"{r['yc'][i]:.6f}",
                        f"{r['thick'][i]:.6f}",
                    ]
                )
        QMessageBox.information(self, "저장 완료", f"CSV 파일이 저장되었습니다:\n{path}")


def auto_build_exe(force=True):
    """코드를 그냥 Run(.py 실행)하면 dist/AirfoilExtractor.exe를 자동 생성"""
    if getattr(sys, "frozen", False):
        return None
    script_path = os.path.abspath(__file__)
    work_dir = os.path.dirname(script_path)
    dist_dir = os.path.join(work_dir, "dist")
    exe_name = "AirfoilExtractor.exe" if sys.platform == "win32" else "AirfoilExtractor"
    exe_path = os.path.join(dist_dir, exe_name)

    if os.path.exists(exe_path) and not force:
        print(f"[안내] 이미 생성된 실행 파일이 존재합니다: {exe_path}")
        return exe_path

    print("=" * 65)
    print("[자동 빌드 시작] 단일 실행 파일(.exe)을 생성하고 있습니다 (약 1~2분 소요)...")
    print("=" * 65)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--noconsole",
        "--clean",
        "--name",
        "AirfoilExtractor",
        "--distpath",
        dist_dir,
        "--workpath",
        os.path.join(work_dir, "build"),
        "--specpath",
        work_dir,
        script_path,
    ]
    subprocess.check_call(cmd)
    print("=" * 65)
    print(f"[자동 빌드 완료] 생성된 파일 위치: {exe_path}")
    print("=" * 65)
    return exe_path


if __name__ == "__main__":
    # .py 스크립트로 직접 Run 했을 때 자동으로 .exe 파일 생성
    if not getattr(sys, "frozen", False) and "--no-build" not in sys.argv:
        try:
            auto_build_exe(force=True)
        except Exception as build_err:
            print(f"[경고] .exe 자동 빌드 중 오류가 발생했습니다: {build_err}")

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
