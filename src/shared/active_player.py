"""Module to detect the active player based on tokens.

Provides token segmentation to recognize whose turn it is using global detection.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple

LAYOUT_RATIOS = {
    "x_left": 0.245, "x_mid_left": 0.3125, "x_mid_right": 0.675, "x_right": 0.775,
    "y_top_left": 0.182, "y_top": 0.309, "y_bottom": 0.691, "y_bottom_right": 0.800
}

def is_background_noisy(img_bgr: np.ndarray) -> bool:
    return bool(img_bgr.std() > 45)

def divide_background(width: int, height: int) -> Dict[str, list]:
    x_left, x_mid_left = int(width * LAYOUT_RATIOS["x_left"]), int(width * LAYOUT_RATIOS["x_mid_left"])
    x_mid_right, x_right = int(width * LAYOUT_RATIOS["x_mid_right"]), int(width * LAYOUT_RATIOS["x_right"])
    y_top_left, y_top = int(height * LAYOUT_RATIOS["y_top_left"]), int(height * LAYOUT_RATIOS["y_top"])
    y_bottom, y_bottom_right = int(height * LAYOUT_RATIOS["y_bottom"]), int(height * LAYOUT_RATIOS["y_bottom_right"])

    return {
        "p3": [(0, 0), (x_right, 0), (x_right, y_top), (x_left, y_top), (x_left, y_top_left), (0, y_top_left)],
        "p4": [(0, y_top_left), (x_left, y_top_left), (x_left, y_top), (x_mid_left, y_top), (x_mid_left, y_bottom), (x_left, y_bottom), (x_left, height), (0, height)],
        "p2": [(x_right, 0), (width, 0), (width, y_bottom_right), (x_right, y_bottom_right), (x_right, y_bottom), (x_mid_right, y_bottom), (x_mid_right, y_top), (x_right, y_top)],
        "p1": [(x_left, y_bottom), (x_right, y_bottom), (x_right, y_bottom_right), (width, y_bottom_right), (width, height), (x_left, height)]
    }

def find_yellow_token(
    img_bgr: np.ndarray, 
    hue_range=(22, 28), sat_min=100, val_min=100,
    dp=1.5, min_dist=50, param1=50, param2=30, min_radius=60, max_radius=100,
    yellow_threshold=0.7, saturation_threshold=0.2
) -> List[Tuple[int, int]]:
    """Scans the whole image and returns a list of (x, y) center points for yellow tokens."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_yellow, upper_yellow = np.array([hue_range[0], sat_min, val_min]), np.array([hue_range[1], 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    
    circles = cv2.HoughCircles(
        cleaned, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius
    )
    
    centers = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            circle_mask = np.zeros_like(cleaned)
            cv2.circle(circle_mask, center, radius, 255, -1)
            
            yellow_in_circle = cv2.bitwise_and(cleaned, circle_mask)
            yellow_ratio = np.sum(yellow_in_circle > 0) / np.sum(circle_mask > 0) if np.sum(circle_mask > 0) > 0 else 0
            
            avg_saturation = np.mean(hsv[:, :, 1][circle_mask > 0]) / 255.0 if np.any(circle_mask > 0) else 0
            
            if yellow_ratio >= yellow_threshold and avg_saturation >= saturation_threshold:
                centers.append(center)
                
    return centers

def find_black_token(img_bgr: np.ndarray) -> List[Tuple[int, int]]:
    """Scans the whole image and returns a list of (x, y) center points for black tokens."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((11, 11), np.uint8)
    cleaned = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for contour in contours:
        if cv2.contourArea(contour) < 35000:
            continue
            
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        if 4 <= len(approx) <= 8 and 0.55 < aspect_ratio < 1.65:
            # Calculate the center of the rectangle using moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
                
    return centers

def detect_active_player(img_bgr: np.ndarray) -> str:
    """
    Detects which player's turn it is based on token position.
    Returns: "p1", "p2", "p3", "p4", or "unknown"
    """
    h, w = img_bgr.shape[:2]
    
    # 1. Detect all valid markers in the entire image first
    if is_background_noisy(img_bgr):
        marker_centers = find_yellow_token(img_bgr)
    else:
        marker_centers = find_black_token(img_bgr)
        
    if not marker_centers:
        return "unknown"
        
    # 2. Map the found coordinates to a sector
    polygons = divide_background(w, h)
    
    for cx, cy in marker_centers:
        for sector_name, polygon in polygons.items():
            pts = np.array(polygon, dtype=np.int32)
            
            # pointPolygonTest returns >= 0 if the point is inside or exactly on the polygon edge
            if cv2.pointPolygonTest(pts, (cx, cy), measureDist=False) >= 0:
                return sector_name
                
    return "unknown"