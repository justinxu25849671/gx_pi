"""Fit top-face hypotheses to original-image edges, never to an invented disk."""
from __future__ import annotations

import math
import cv2
import numpy as np


DEFAULTS = {
    "top_canny_low": 45, "top_canny_high": 120,
    "top_min_axis_ratio": 0.35, "top_min_area_ratio": 0.12,
    "top_max_area_ratio": 1.15, "top_edge_tolerance_px": 2.5,
    "top_min_edge_support": 0.78, "top_min_arc_coverage": 0.75,
    "top_border_margin_px": 3, "top_fit_trials": 64,
}


def fit_top(frame, contour, config):
    """Return ellipse, rejection reason, metrics, and original-resolution crop edges."""
    cfg = {**DEFAULTS, **config}
    x, y, w, h = cv2.boundingRect(contour)
    height, width = frame.shape[:2]
    margin = cfg["top_border_margin_px"]
    if x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin:
        return None, "目标不完整：接触画面边缘", {}, None
    pad = 4
    x0, y0 = max(0, x-pad), max(0, y-pad)
    x1, y1 = min(width, x+w+pad), min(height, y+h+pad)
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0),
                      cfg["top_canny_low"], cfg["top_canny_high"])
    # Luminance alone can hide a rim between saturated surfaces.
    value = np.max(crop, axis=2)
    edges |= cv2.Canny(cv2.GaussianBlur(value, (3, 3), 0),
                       cfg["top_canny_low"], cfg["top_canny_high"])
    region = np.zeros_like(edges)
    local_contour = contour - np.array([[[x0, y0]]], dtype=np.int32)
    cv2.drawContours(region, [local_contour], -1, 255, -1)
    region = cv2.dilate(region, np.ones((7, 7), np.uint8))
    edges = cv2.bitwise_and(edges, region)
    diagnostic = (x0, y0, edges)
    yy, xx = np.nonzero(edges)
    points = np.column_stack((xx, yy)).astype(np.float32)
    if len(points) > 4000:
        points = points[np.linspace(0, len(points)-1, 4000).astype(int)]
    if len(points) < 20:
        return None, "边缘不足", {}, diagnostic
    distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    hypotheses = [c.reshape(-1, 2)[::max(1, len(c)//1000)].astype(np.float32)
                  for c in sorted(contours, key=len, reverse=True)[:32] if len(c) >= 12]
    # Partial upper arcs give seeds even when the side wall joins the rim.
    for fraction in (0.4, 0.55, 0.7, 0.85, 1.0):
        arc = points[points[:, 1] <= y-y0 + h*fraction]
        if len(arc) >= 12:
            hypotheses.append(arc)
    rng = np.random.default_rng(0)
    upper = points[points[:, 1] < y-y0+h*0.8]
    if len(upper) >= 8:
        for _ in range(int(cfg["top_fit_trials"])):
            hypotheses.append(upper[rng.choice(len(upper), 8, replace=False)])
    angles = np.linspace(0, 2*math.pi, 144, endpoint=False)
    area = cv2.contourArea(contour)
    best = None
    best_score = -1.0
    best_metrics = {}
    reason = "尺寸或轴比不符"
    for pts in hypotheses:
        try:
            ellipse = cv2.fitEllipse(pts)
        except cv2.error:
            continue
        (cx, cy), (a, b), angle = ellipse
        if not np.isfinite([cx, cy, a, b, angle]).all() or min(a, b) <= 4:
            continue
        major, minor = max(a, b), min(a, b)
        area_ratio = math.pi*a*b/4/max(area, 1)
        if not (0.55*w <= major <= 1.25*w and minor/major >= cfg["top_min_axis_ratio"]
                and cfg["top_min_area_ratio"] <= area_ratio <= cfg["top_max_area_ratio"]
                and y-y0 <= cy <= y-y0+h*0.7 and x-x0 <= cx <= x-x0+w):
            continue
        rad = math.radians(angle)
        ex = cx + a/2*np.cos(angles)*math.cos(rad) - b/2*np.sin(angles)*math.sin(rad)
        ey = cy + a/2*np.cos(angles)*math.sin(rad) + b/2*np.sin(angles)*math.cos(rad)
        if ex.min() < 0 or ey.min() < 0 or ex.max() >= edges.shape[1]-1 or ey.max() >= edges.shape[0]-1:
            continue
        ix, iy = np.rint(ex).astype(int), np.rint(ey).astype(int)
        supported = distance[iy, ix] <= cfg["top_edge_tolerance_px"]
        support = float(supported.mean())
        coverage = float((supported.reshape(24, 6).mean(axis=1) >= 0.5).mean())
        inside = float((region[iy, ix] > 0).mean())
        normalized = np.column_stack(((points[:, 0]-cx)*math.cos(rad)+(points[:, 1]-cy)*math.sin(rad),
                                       -(points[:, 0]-cx)*math.sin(rad)+(points[:, 1]-cy)*math.cos(rad)))
        radius = np.sqrt((normalized[:, 0]/(a/2))**2 + (normalized[:, 1]/(b/2))**2)
        residual = float(np.median(np.abs(radius[np.abs(radius-1) < 0.15]-1))) if np.any(np.abs(radius-1)<0.15) else 1.
        rim = normalized[np.abs(radius-1) < 0.08] / np.array([a/2, b/2])
        circularity = 0.0
        if len(rim) >= 5:
            hull = cv2.convexHull(rim.astype(np.float32))
            (_, _), rr = cv2.minEnclosingCircle(hull)
            circularity = cv2.contourArea(hull)/max(math.pi*rr*rr, 1e-6)
        metrics = {"circularity": circularity,"edge_support": support, "arc_coverage": coverage,
                   "axis_ratio": minor/major, "area_ratio": area_ratio, "fit_residual": residual}
        score = support + coverage - residual - 0.05*cy/max(h, 1)
        valid = (support >= cfg["top_min_edge_support"] and coverage >= cfg["top_min_arc_coverage"] and inside >= 0.95
                 and circularity >= float(cfg.get("min_circularity", 0.55)))
        if valid:
            score += 3
        if score > best_score:
            best_score, best_metrics = score, metrics
            best = ((cx+x0, cy+y0), (a, b), angle)
            reason = "" if valid else "边缘贴合不足" if support < cfg["top_min_edge_support"] else "圆形筛选不符" if circularity < float(cfg.get("min_circularity", 0.55)) else "圆弧覆盖不足"
    return best, reason, best_metrics, diagnostic
