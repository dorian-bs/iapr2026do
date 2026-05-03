from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog as skimage_hog

from uno_vision.image_ops import letterbox_bgr


IMG_SIZE = 128
FOURIER_COEFFS = 32
SPECIAL_COLOR_TOKEN = "__special__"
SPECIAL_CARDS = {"wild", "draw_4"}


def split_card_label(card: str) -> tuple[str, str]:
    if card in SPECIAL_CARDS:
        return SPECIAL_COLOR_TOKEN, card
    if "_" in card:
        return card.split("_", 1)
    return SPECIAL_COLOR_TOKEN, card


def compose_card_label(color: str, rank: str) -> str:
    if rank in SPECIAL_CARDS or color == SPECIAL_COLOR_TOKEN:
        return rank
    return f"{color}_{rank}"


def extract_color_features(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 40).astype(np.uint8) * 255
    h_hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [16], [0, 256]).flatten()
    feat = np.concatenate([h_hist, s_hist])
    return feat / (feat.sum() + 1e-6)


def extract_hog_features(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return skimage_hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


def extract_fourier_features(img_bgr: np.ndarray, n_coeffs: int = FOURIER_COEFFS) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw[:8, :] = 0
    bw[-8:, :] = 0
    bw[:, :8] = 0
    bw[:, -8:] = 0
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(n_coeffs, dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt[:, 0, :].astype(np.float32)
    if len(pts) < max(10, n_coeffs + 1):
        return np.zeros(n_coeffs, dtype=np.float32)
    z = pts[:, 0] + 1j * pts[:, 1]
    z = z - np.mean(z)
    fft = np.fft.fft(z)
    mag = np.abs(fft)[1:n_coeffs + 1]
    if len(mag) < n_coeffs:
        mag = np.pad(mag, (0, n_coeffs - len(mag)))
    mag = mag / (mag[0] + 1e-6)
    return mag.astype(np.float32)


def extract_rank_features(img_bgr: np.ndarray) -> np.ndarray:
    hog_vec = extract_hog_features(img_bgr)
    fourier_vec = extract_fourier_features(img_bgr)
    return np.concatenate([hog_vec, fourier_vec])


def extract_features_from_path(path: str, size: int = IMG_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img_lb = letterbox_bgr(img_bgr, size=size)
    return extract_color_features(img_lb), extract_rank_features(img_lb), img_lb


letterbox_cv2 = letterbox_bgr
extract_color_feat = extract_color_features
extract_hog_feat = extract_hog_features
extract_fourier_feat = extract_fourier_features
extract_rank_feat = extract_rank_features