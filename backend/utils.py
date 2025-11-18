"""Utility functions for drawing and visualization."""
import cv2
import numpy as np
from backend.config import RENDER_WALLS_AS_BOXES, DISABLE_WALL_LABELS

BLUE = (255, 0, 0)  # BGR - pure blue for walls


def draw_box(
    img: np.ndarray,
    xyxy,
    *,
    color=(0, 255, 0),
    mode="filled",          # "filled" | "outline"
    alpha=0.25,             # fill opacity
    thickness=2,            # outline thickness
    label: str | None = None
):
    """Generic box drawing function used for all classes."""
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return img

    if mode == "filled":
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = x1 + 6, max(y1 - 6, th + 6)
        tag = img.copy()
        cv2.rectangle(tag, (tx-4, ty-th-6), (tx-4+tw+8, ty+4), (255,255,255), -1)
        cv2.addWeighted(tag, 0.35, img, 0.65, 0, img)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
    return img


def draw_detections(
    image_bgr: np.ndarray,
    detections: list[dict],
    *,
    wall_box_mode="filled",       # "filled" or "outline"
    wall_box_alpha=0.25,
    wall_box_thickness=2,
    show_labels=False,
    line_style="solid",  # "solid", "dashed", "dotted" - for outline mode
):
    """
    Unified detection renderer: walls as blue boxes, other classes with their colors.
    Walls never show labels. No legacy wall overlay.
    """
    img = image_bgr.copy()

    # 1) Walls as BLUE boxes (no legacy overlay, no wall labels)
    for det in detections:
        if det.get("cls_name") != "wall":
            continue
        draw_box(
            img,
            det["xyxy"],
            color=BLUE,
            mode=wall_box_mode,
            alpha=wall_box_alpha,
            thickness=wall_box_thickness,
            label=None,                # suppress "Wall ..." text
        )

    # 2) Other classes (keep their colors/labels as-is)
    for det in detections:
        if det.get("cls_name") == "wall":
            continue
        cls_name = det.get("cls_name")
        xyxy = det.get("xyxy")
        if not xyxy or len(xyxy) < 4:
            continue
        conf = det.get("conf")
        label = f"{cls_name.title()} {conf:.2f}" if (show_labels and conf is not None) else (cls_name.title() if show_labels else None)

        # Colors for other classes
        if cls_name == "door":
            color = (0, 255, 0)  # Green
        elif cls_name == "window":
            color = (0, 200, 255)  # Orange/cyan
        else:
            color = (128, 0, 255)  # Purple for unknown

        # For doors/windows, use outline mode with line style support
        # Note: draw_box doesn't support line styles yet, so we use solid for now
        draw_box(img, xyxy, color=color, mode="outline", thickness=wall_box_thickness, label=label)

    return img

