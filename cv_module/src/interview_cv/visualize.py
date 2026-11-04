"""Optional drawing of landmarks and observation text onto frames (debugging / demo only)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .pipeline import FrameAnalysis

_GREEN = (80, 220, 80)
_RED = (60, 60, 230)
_YELLOW = (40, 200, 240)
_WHITE = (240, 240, 240)
_CYAN = (230, 200, 60)

# Pose connections for the upper body only (MediaPipe Pose indices).
_UPPER_BODY_CONNECTIONS = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (0, 11), (0, 12))


def draw_face_landmarks(image: np.ndarray, landmarks: np.ndarray, color=_GREEN, radius: int = 1) -> None:
    h, w = image.shape[:2]
    for x, y in landmarks[:, :2]:
        cv2.circle(image, (int(x * w), int(y * h)), radius, color, -1, lineType=cv2.LINE_AA)


def draw_pose_landmarks(image: np.ndarray, landmarks: np.ndarray, color=_CYAN) -> None:
    h, w = image.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks[:, :2]]
    for a, b in _UPPER_BODY_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(image, pts[a], pts[b], color, 2, lineType=cv2.LINE_AA)
    for i in (0, 11, 12):
        if i < len(pts):
            cv2.circle(image, pts[i], 4, color, -1, lineType=cv2.LINE_AA)


def draw_overlay(image_bgr: np.ndarray, analysis: FrameAnalysis, landmarks: bool = True) -> np.ndarray:
    """Return an annotated copy of the frame."""
    img = image_bgr.copy()
    obs = analysis.observation
    h, w = img.shape[:2]
    scale = max(0.5, min(h, w) / 800.0)  # keep text legible on both 480p and 4K frames
    line = int(36 * scale)
    y = line

    def put(text: str, color=_WHITE) -> None:
        nonlocal y
        thickness = max(1, int(round(scale)))
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line

    put(f"t={obs.timestamp_s:.2f}s  frame {obs.frame_index}")

    if obs.face and obs.face.present and obs.face.bbox:
        x0, y0, x1, y1 = obs.face.bbox.to_pixels(w, h)
        cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, 2)
        if landmarks and analysis.face_landmarks is not None:
            draw_face_landmarks(img, analysis.face_landmarks)
        hp = obs.face.head_pose
        if hp:
            put(f"head yaw {hp.yaw_deg:+.0f} pitch {hp.pitch_deg:+.0f} roll {hp.roll_deg:+.0f}")
    else:
        put("face: not detected", _RED)

    if obs.gaze:
        color = _GREEN if obs.gaze.state == "camera" else (_RED if obs.gaze.state == "away" else _YELLOW)
        put(f"gaze: {obs.gaze.state} ({obs.gaze.direction})", color)

    if obs.expression:
        p = obs.expression.probabilities.get(obs.expression.label, 0.0)
        put(f"expression: {obs.expression.label} [{obs.expression.group}] {p:.2f}")

    if obs.posture:
        if obs.posture.upper_body_visible:
            if landmarks and analysis.pose_landmarks is not None:
                draw_pose_landmarks(img, analysis.pose_landmarks)
            flags = ", ".join(obs.posture.flags) or "ok"
            put(f"posture: tilt {obs.posture.shoulder_tilt_deg:+.0f} offset {obs.posture.head_offset_ratio:+.2f} -> {flags}",
                _GREEN if not obs.posture.flags else _YELLOW)
        else:
            put("posture: upper body not visible", _YELLOW)

    for err in obs.errors:
        put(f"error: {err}", _RED)
    return img
