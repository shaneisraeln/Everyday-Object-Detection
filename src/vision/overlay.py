"""Renders minimalist, high-contrast HUD bounding boxes and active target reticles onto frames."""

import cv2
import numpy as np
from src.schemas import Detection

# Minimalist, high-contrast palette
PALETTE = [
    (56, 189, 248),   # Sky blue
    (167, 139, 250),  # Soft purple
    (251, 191, 36),   # Warm amber
    (6, 182, 212),    # Clean cyan
    (244, 114, 182),  # Soft pink
    (251, 146, 60),   # Vibrant orange
    (99, 102, 241),   # Indigo
]

# High-visibility Emerald Green for targeted/active object
ACTIVE_ACCENT = (16, 185, 129)


def render_detections_overlay(
    image: np.ndarray,
    detections: list[Detection],
    active_index: int | None = None,
    active_label: str | None = None,
    suppress_person_when_object_present: bool = True,
) -> np.ndarray:
    """Draw sleek bounding boxes, corner brackets, and minimalist tags onto the frame.
    
    Args:
        image: Source image numpy array (RGB or BGR).
        detections: List of Detection items.
        active_index: Index of currently selected detection (if any).
        active_label: Label of currently active object (used to highlight matches).
        suppress_person_when_object_present: If True and non-person objects exist, omit the person box.
        
    Returns:
        Annotated image numpy array.
    """
    if image is None or image.size == 0 or not detections:
        return image.copy() if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    # Filter out person bbox if everyday objects are present
    display_dets = detections
    has_non_person = any(d.label.lower() != "person" for d in detections)
    if suppress_person_when_object_present and has_non_person:
        display_dets = [d for d in detections if d.label.lower() != "person"]

    if not display_dets:
        display_dets = detections

    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Pre-sort so non-active are drawn first and active is drawn on top
    def get_sort_key(item: tuple[int, Detection]) -> int:
        idx, det = item
        is_active = (
            (active_index is not None and idx == active_index)
            or (active_label and det.label.lower() == active_label.lower())
        )
        return 1 if is_active else 0

    sorted_enumerated = sorted(list(enumerate(display_dets)), key=get_sort_key)

    # 1. First pass: Draw bounding boxes and corner brackets
    for idx, det in sorted_enumerated:
        x1, y1, x2, y2 = det.bbox_xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        is_active = (
            (active_index is not None and idx == active_index)
            or (active_label and det.label.lower() == active_label.lower())
        )

        accent = ACTIVE_ACCENT if is_active else PALETTE[det.class_id % len(PALETTE)]
        box_thickness = 2 if is_active else 1

        # Main box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), accent, box_thickness, lineType=cv2.LINE_AA)

        # Sleek corner accents (4 corners)
        bw = x2 - x1
        bh = y2 - y1
        corner_len = max(8, min(24, int(min(bw, bh) * 0.2)))
        corner_thick = 3 if is_active else 2

        # Top-left corner
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), accent, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), accent, corner_thick, cv2.LINE_AA)
        # Top-right corner
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), accent, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), accent, corner_thick, cv2.LINE_AA)
        # Bottom-left corner
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), accent, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), accent, corner_thick, cv2.LINE_AA)
        # Bottom-right corner
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), accent, corner_thick, cv2.LINE_AA)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), accent, corner_thick, cv2.LINE_AA)

    # 2. Second pass: Draw translucent badge backdrops
    overlay = annotated.copy()
    badge_rects = []

    for idx, det in sorted_enumerated:
        x1, y1, x2, y2 = det.bbox_xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        is_active = (
            (active_index is not None and idx == active_index)
            or (active_label and det.label.lower() == active_label.lower())
        )

        tag = f"[{idx + 1}] {det.label} {det.confidence:.0%}"
        if is_active:
            tag += " [ACTIVE]"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(tag, font, font_scale, font_thick)

        pad_x, pad_y = 6, 4
        badge_w = tw + (pad_x * 2) + 6
        badge_h = th + (pad_y * 2) + 2

        if y1 - badge_h >= 2:
            by1 = y1 - badge_h
            by2 = y1
        else:
            by1 = y1
            by2 = min(h, y1 + badge_h)

        bx1 = x1
        bx2 = min(w, x1 + badge_w)

        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (15, 15, 18), -1)
        badge_rects.append((idx, det, tag, bx1, by1, bx2, by2, th, pad_x, pad_y, is_active))

    cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)

    # 3. Third pass: Draw text and accent indicator bars on top of blended backdrops
    for (idx, det, tag, bx1, by1, bx2, by2, th, pad_x, pad_y, is_active) in badge_rects:
        accent = ACTIVE_ACCENT if is_active else PALETTE[det.class_id % len(PALETTE)]

        # Thin accent bar on left edge of badge
        cv2.rectangle(annotated, (bx1, by1), (bx1 + 3, by2), accent, -1)

        # Anti-aliased text
        text_y = by1 + th + pad_y
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        font_thick = 1
        cv2.putText(
            annotated,
            tag,
            (bx1 + pad_x + 3, text_y),
            font,
            font_scale,
            (250, 250, 250),
            font_thick,
            cv2.LINE_AA,
        )

    return annotated
