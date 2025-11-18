"""Overlay rendering for detections and rooms."""
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from backend.geometry import calculate_area_sf
from backend.utils import draw_detections
from backend.config import DISABLE_LEGACY_WALL_OVERLAY

# Kill-switch for legacy wall overlay (use config) - ALWAYS OFF
USE_LEGACY_WALL_OVERLAY_DEFAULT = True  # Set to True to disable legacy (inverted logic)
DISABLE_LEGACY_WALL_OVERLAY = True  # Hard disable

logger = logging.getLogger(__name__)

# --- continuous-wall helpers ---
import numpy as _np
import cv2 as _cv


def _clip_box(_b, W, H):
    x1, y1, x2, y2 = [int(round(v)) for v in _b]
    x1 = max(0, min(x1, W-1))
    x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1))
    y2 = max(0, min(y2, H-1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _draw_boxes_to_mask(mask, boxes, thick=1):
    H, W = mask.shape[:2]
    for b in boxes or []:
        x1, y1, x2, y2 = _clip_box(b, W, H)
        _cv.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=max(1, thick))


def _continuous_wall_mask(image_shape, walls_xyxy, mask_dilate_px=4, gap_close_px=12):
    """
    Build a binary mask of walls with minimal dilation and oriented closings
    so boxes become one continuous stripe, not a blocky fill.
    """
    H, W = image_shape[:2]
    m = _np.zeros((H, W), _np.uint8)
    _draw_boxes_to_mask(m, walls_xyxy, thick=1)
    
    # First, connect nearby aligned segments with larger gap closing
    g = max(3, int(gap_close_px))
    k_h = _cv.getStructuringElement(_cv.MORPH_RECT, (g * 2, 1))
    k_v = _cv.getStructuringElement(_cv.MORPH_RECT, (1, g * 2))
    m_h = _cv.morphologyEx(m, _cv.MORPH_CLOSE, k_h)
    m_v = _cv.morphologyEx(m, _cv.MORPH_CLOSE, k_v)
    m = _cv.bitwise_or(m_h, m_v)
    
    # Then dilate to ensure connectivity
    if mask_dilate_px > 0:
        k = _cv.getStructuringElement(_cv.MORPH_RECT, (mask_dilate_px, mask_dilate_px))
        m = _cv.dilate(m, k, iterations=1)
        # Additional small dilation for better connectivity
        m = _cv.dilate(m, k, iterations=1)
    
    # Final closing pass to ensure continuous lines
    k_h2 = _cv.getStructuringElement(_cv.MORPH_RECT, (g, 1))
    k_v2 = _cv.getStructuringElement(_cv.MORPH_RECT, (1, g))
    m_h2 = _cv.morphologyEx(m, _cv.MORPH_CLOSE, k_h2)
    m_v2 = _cv.morphologyEx(m, _cv.MORPH_CLOSE, k_v2)
    m = _cv.bitwise_or(m_h2, m_v2)
    
    return m


def _skeletonize(mask):
    """
    Return a 1-pixel skeleton of the binary mask.
    Use ximgproc.thinning if available; otherwise fall back to morphological skeleton.
    """
    if hasattr(_cv, "ximgproc") and hasattr(_cv.ximgproc, "thinning"):
        return _cv.ximgproc.thinning(mask)  # fast & clean
    # Fallback morphological skeleton
    skel = _np.zeros(mask.shape, _np.uint8)
    element = _cv.getStructuringElement(_cv.MORPH_CROSS, (3, 3))
    done = False
    m = mask.copy()
    while not done:
        open_ = _cv.morphologyEx(m, _cv.MORPH_OPEN, element)
        temp = _cv.subtract(m, open_)
        eroded = _cv.erode(m, element)
        skel = _cv.bitwise_or(skel, temp)
        m = eroded.copy()
        done = (_cv.countNonZero(m) == 0)
    return skel


def _draw_skeleton(img_bgr, skel, color_bgr=(90, 60, 255), thickness=2):
    """
    Draw skeleton as thin continuous lines by extracting its contours.
    Also extends lines to connect nearby endpoints.
    """
    cnts, _ = _cv.findContours(skel, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_NONE)
    
    # Draw all contours
    for c in cnts:
        if len(c) < 2:
            continue
        _cv.polylines(img_bgr, [c], isClosed=False, color=color_bgr, thickness=thickness)
    
    # Extend skeleton lines to connect nearby endpoints
    if len(cnts) > 1:
        endpoints = []
        for c in cnts:
            if len(c) >= 2:
                # Get start and end points
                start = tuple(c[0][0])
                end = tuple(c[-1][0])
                endpoints.append((start, end))
        
        # Connect nearby endpoints (within threshold)
        connect_threshold = 30
        for i, (p1_start, p1_end) in enumerate(endpoints):
            for j, (p2_start, p2_end) in enumerate(endpoints):
                if i >= j:
                    continue
                
                # Check all combinations of endpoints
                for ep1 in [p1_start, p1_end]:
                    for ep2 in [p2_start, p2_end]:
                        dist = _np.sqrt((ep1[0] - ep2[0])**2 + (ep1[1] - ep2[1])**2)
                        if dist <= connect_threshold and dist > 0:
                            _cv.line(img_bgr, ep1, ep2, color_bgr, thickness)


# --- end helpers ---


# Color palette for classes
CLASS_COLORS = {
    "wall": (100, 100, 255),  # Blue
    "door": (100, 255, 100),  # Green
    "window": (100, 100, 255),  # Blue
    "room": (200, 200, 200),  # Light gray for outlines
}

# Room fill colors (distinct palette)
ROOM_COLORS = [
    (255, 182, 193),  # Light pink
    (144, 238, 144),  # Light green
    (173, 216, 230),  # Light blue
    (255, 228, 181),  # Moccasin
    (221, 160, 221),  # Plum
    (255, 218, 185),  # Peach
    (176, 224, 230),  # Powder blue
    (255, 192, 203),  # Pink
    (152, 251, 152),  # Pale green
    (255, 250, 205),  # Lemon chiffon
]


def render_overlay(
    original_image: np.ndarray,
    detections: Dict[str, List[Dict]],
    rooms: List[Dict],
    settings: Dict,
) -> np.ndarray:
    """
    Render overlay on original image.
    
    Args:
        original_image: Original image as numpy array (BGR)
        detections: Dictionary of detections by class
        rooms: List of room dictionaries
        settings: Dictionary with overlay settings:
            - show_labels: bool
            - fill_rooms: bool
            - outline_interior: bool
            - outline_thickness: int
            - overlay_style: str ("simple" or "detailed")
            - pixels_per_foot: Optional[float]
    
    Returns:
        Overlay image as numpy array (BGR)
    """
    overlay = original_image.copy()
    
    # Helper to parse flag values
    def _parse_flag(value, default: bool = False) -> bool:
        """Parse a flag value (string or bool) to boolean."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "on"}
    
    show_labels = _parse_flag(settings.get("show_labels"), True)
    show_walls = _parse_flag(settings.get("show_walls"), True)
    show_doors = _parse_flag(settings.get("show_doors"), True)
    show_windows = _parse_flag(settings.get("show_windows"), True)
    
    # Room rendering flags - use explicit names, fall back to legacy names
    show_rooms_fill_raw = settings.get("show_rooms_fill") or settings.get("fill_rooms")
    show_room_outlines_raw = settings.get("show_room_outlines") or settings.get("outline_interior")
    
    show_rooms_fill = _parse_flag(show_rooms_fill_raw, False)
    show_room_outlines = _parse_flag(show_room_outlines_raw, False)
    
    outline_thickness = settings.get("outline_thickness", 2)
    line_thickness = settings.get("line_thickness", outline_thickness)  # Use new line_thickness if available
    line_style = settings.get("line_style", "solid")  # "solid", "dashed", "dotted"
    overlay_style = settings.get("overlay_style", "simple")
    pixels_per_foot = settings.get("pixels_per_foot")
    
    # Render rooms using the unified helper
    # This ensures fill and outlines are completely separate
    room_mask = None  # We don't have a room mask, only polygons
    room_color = (255, 0, 255)  # Magenta in BGR for outlines
    fill_alpha = settings.get("room_fill_alpha", 0.35)  # Opacity of room fills
    
    overlay = render_rooms_layer(
        overlay_bgr=overlay,
        room_mask=room_mask,
        room_polys=rooms if rooms else [],
        show_fill=show_rooms_fill,
        show_outlines=show_room_outlines,
        room_color=room_color,
        outline_thickness=line_thickness,  # Use line_thickness instead of outline_thickness
        fill_alpha=fill_alpha,
        line_style=line_style,  # Pass line_style for dashed/dotted support
    )
    
    # Log the actual boolean values being used
    logger.info(f"[OVERLAY] Room rendering - show_rooms_fill: {show_rooms_fill}, show_room_outlines: {show_room_outlines}")
    if not show_rooms_fill and not show_room_outlines:
        logger.debug(f"[OVERLAY] Both room flags are False - skipping all room rendering")
    elif show_rooms_fill and show_room_outlines:
        logger.debug(f"[OVERLAY] Drawing room fills + outlines: {len(rooms) if rooms else 0} rooms")
    elif show_rooms_fill:
        logger.debug(f"[OVERLAY] Drawing room fills only: {len(rooms) if rooms else 0} rooms, alpha={fill_alpha}")
    elif show_room_outlines:
        logger.debug(f"[OVERLAY] Drawing room outlines only: {len(rooms) if rooms else 0} rooms, thickness={outline_thickness}")
    
    H, W = overlay.shape[:2]
    
    # Convert detections to flat list format for unified draw_detections
    # It expects: [{"cls_name": "wall", "xyxy": [x1,y1,x2,y2], "conf": ...}, ...]
    detections_list = []
    for class_name, class_detections in detections.items():
        for det in class_detections:
            bbox = det["bbox"]
            confidence = det.get("confidence", 0.0)
            
            # Normalize bbox to pixel coordinates
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            normalized = max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0
            if normalized:
                if x2 <= 1.0 and y2 <= 1.0:
                    # xywh normalized
                    x, y, w, h = x1*W, y1*H, x2*W, y2*H
                    X1, Y1, X2, Y2 = x, y, x+w, y+h
                else:
                    # xyxy normalized
                    X1, Y1, X2, Y2 = x1*W, y1*H, x2*W, y2*H
            else:
                # Already pixels - check if xywh
                if x2 <= 2 or y2 <= 2:
                    x, y, w, h = x1, y1, x2, y2
                    X1, Y1, X2, Y2 = x, y, x+w, y+h
                else:
                    X1, Y1, X2, Y2 = x1, y1, x2, y2
            
            # Clip and order
            X1 = max(0, min(int(round(X1)), W-1))
            Y1 = max(0, min(int(round(Y1)), H-1))
            X2 = max(0, min(int(round(X2)), W-1))
            Y2 = max(0, min(int(round(Y2)), H-1))
            if X2 < X1:
                X1, X2 = X2, X1
            if Y2 < Y1:
                Y1, Y2 = Y2, Y1
            
            detections_list.append({
                "cls_name": class_name,
                "xyxy": [X1, Y1, X2, Y2],
                "conf": confidence,
            })
    
    # Filter detections based on layer visibility
    filtered_detections_list = []
    for det in detections_list:
        cls_name = det.get("cls_name", "")
        if cls_name == "wall" and not show_walls:
            continue
        if cls_name == "door" and not show_doors:
            continue
        if cls_name == "window" and not show_windows:
            continue
        filtered_detections_list.append(det)
    
    # Draw walls using exact visual rectangles (matching detection box dimensions exactly)
    # This is decoupled from the thick mask used for room finding
    wall_rects_visual = settings.get("wall_rects_visual")
    
    if wall_rects_visual and show_walls:
        # Draw exact rectangles for walls (matching detection boxes exactly - no padding)
        # Use filled rectangles to match exact wall thickness
        wall_color = (255, 0, 0)  # Blue in BGR (appears as red in RGB)
        
        for (x1, y1, x2, y2) in wall_rects_visual:
            # Draw filled rectangle to match exact wall dimensions
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                wall_color,
                thickness=-1,  # Filled rectangle to match exact wall thickness
            )
        
        # Filter out wall detections from the list since we're drawing from rectangles
        filtered_detections_list = [d for d in filtered_detections_list if d.get("cls_name") != "wall"]
    else:
        # Filter out wall detections from the list since we're not drawing them
        filtered_detections_list = [d for d in filtered_detections_list if d.get("cls_name") != "wall"]
    
    # Draw windows using exact visual rectangles (matching detection box dimensions exactly)
    # Draw as outlines that match the exact window dimensions
    if show_windows:
        window_dets = [d for d in filtered_detections_list if d.get("cls_name") == "window"]
        if window_dets:
            window_color = (0, 200, 255)  # Orange/cyan in BGR
            # Use line_thickness for window outlines
            window_thickness = max(1, line_thickness)
            for det in window_dets:
                xyxy = det.get("xyxy")
                if not xyxy or len(xyxy) < 4:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                # Draw exact rectangle matching detection box dimensions
                # For thickness > 1, OpenCV draws outline centered on boundary, so we shrink
                # the box by half the thickness to keep the outer edge at the exact dimensions
                if window_thickness > 1:
                    shrink = window_thickness // 2
                    x1_adj = max(0, x1 + shrink)
                    y1_adj = max(0, y1 + shrink)
                    x2_adj = min(overlay.shape[1] - 1, x2 - shrink)
                    y2_adj = min(overlay.shape[0] - 1, y2 - shrink)
                    if x2_adj > x1_adj and y2_adj > y1_adj:
                        cv2.rectangle(
                            overlay,
                            (x1_adj, y1_adj),
                            (x2_adj, y2_adj),
                            window_color,
                            thickness=window_thickness,
                            lineType=cv2.LINE_AA,
                        )
                else:
                    # For thickness=1, draw directly at exact coordinates
                    cv2.rectangle(
                        overlay,
                        (x1, y1),
                        (x2, y2),
                        window_color,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
            # Remove windows from filtered list since we're drawing them separately
            filtered_detections_list = [d for d in filtered_detections_list if d.get("cls_name") != "window"]
    
    # Draw doors (filtered based on visibility)
    if filtered_detections_list:
        wall_box_mode = settings.get("wall_box_mode", "filled")
        wall_box_alpha = settings.get("wall_box_alpha", 0.25)
        wall_box_thickness = settings.get("wall_box_thickness", 2)
        
        overlay = draw_detections(
            overlay,
            filtered_detections_list,
            wall_box_mode=wall_box_mode,
            wall_box_alpha=wall_box_alpha,
            wall_box_thickness=line_thickness,  # Use line_thickness for consistency
            show_labels=show_labels,
            line_style=line_style,  # Pass line_style for dashed/dotted support
        )
    
    # Draw room labels
    if show_labels and rooms:
        # Sort rooms by area for consistent coloring
        sorted_rooms = sorted(rooms, key=lambda r: r.get("area_px", 0), reverse=True)
        
        for i, room in enumerate(sorted_rooms):
            centroid = room.get("centroid", [0, 0])
            cx, cy = int(centroid[0]), int(centroid[1])
            
            label_parts = [f"Room {i + 1}"]
            
            if overlay_style == "detailed":
                # Prefer square feet if available, otherwise show px²
                area_sf = room.get("area_sf") or room.get("area_sqft")
                if area_sf is not None:
                    label_parts.append(f"{area_sf:.1f} ft²")
                else:
                    area_px = room.get("area_px", 0) or room.get("area_px2", 0)
                    if area_px > 0:
                        label_parts.append(f"{area_px:.0f} px²")
            
            label_text = " | ".join(label_parts)
            room_color = ROOM_COLORS[i % len(ROOM_COLORS)]
            draw_label(overlay, label_text, (cx, cy), room_color, bg_color=(0, 0, 0))
    
    return overlay


def draw_room_fills(image: np.ndarray, rooms: List[Dict], fill_alpha: float = 0.4) -> np.ndarray:
    """
    Fill rooms with distinct solid colors using mask-based approach.
    Each room gets a different color from the palette.
    FILL ONLY - absolutely NO borders/outlines.
    Uses proper alpha blending to avoid edge artifacts.
    """
    if not rooms or len(rooms) == 0:
        return image
    
    h, w = image.shape[:2]
    result = image.copy().astype(float)
    
    # Sort rooms by area to ensure consistent coloring
    sorted_rooms = sorted(rooms, key=lambda r: r.get("area_px", 0), reverse=True)
    
    for i, room in enumerate(sorted_rooms):
        polygon = room.get("polygon")
        if not polygon or len(polygon) < 3:
            continue
        
        # Use distinct color for each room
        color = ROOM_COLORS[i % len(ROOM_COLORS)]
        # Convert RGB to BGR for OpenCV
        bgr_color = np.array([color[2], color[1], color[0]], dtype=np.float32)
        
        # Create mask for this room
        room_mask = np.zeros((h, w), dtype=np.uint8)
        poly_points = np.array(polygon, dtype=np.int32)
        # Fill polygon in mask - NO outline, smooth edges
        cv2.fillPoly(room_mask, [poly_points], 255, lineType=cv2.LINE_AA)
        
        # Apply mask-based fill with proper alpha blending
        mask_f = room_mask.astype(float) / 255.0
        alpha_m = fill_alpha * mask_f[..., None]
        
        # Blend: result = original * (1 - alpha) + color * alpha
        result = result * (1 - alpha_m) + bgr_color * alpha_m
    
    return result.astype(np.uint8)


def draw_mask_fill(
    overlay: np.ndarray,
    mask: np.ndarray,
    color: tuple,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Fill a mask region with a color, blended with the overlay.
    FILL ONLY - no borders/outlines.
    """
    if mask is None or mask.size == 0:
        return overlay
    
    # Create colored overlay
    colored_overlay = overlay.copy()
    colored_overlay[mask > 0] = color
    
    # Blend with original
    result = cv2.addWeighted(overlay, 1 - alpha, colored_overlay, alpha, 0)
    return result


def draw_mask_fill_only(overlay_bgr: np.ndarray, mask: np.ndarray, color=(255, 0, 255), alpha=0.35) -> np.ndarray:
    """
    Draw ROOM FILL ONLY — absolutely NO outlines.
    Uses proper alpha blending to avoid edge artifacts.
    """
    if mask is None:
        return overlay_bgr
    
    # Ensure mask is uint8 0/255
    m = (mask > 0).astype("uint8") * 255
    
    color_arr = np.array(color, dtype=np.uint8)
    color_img = np.zeros_like(overlay_bgr, dtype=np.uint8)
    color_img[:] = color_arr
    
    mask_f = m.astype(float) / 255.0
    alpha_m = alpha * mask_f[..., None]
    
    overlay_bgr = overlay_bgr.astype(float)
    overlay_bgr = overlay_bgr * (1 - alpha_m) + color_img.astype(float) * alpha_m
    
    return overlay_bgr.astype(np.uint8)


def create_room_mask_from_polys(room_polys: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a binary mask from room polygons.
    Returns a uint8 mask where rooms are 255 and background is 0.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for room in room_polys:
        polygon = room.get("polygon")
        if not polygon or len(polygon) < 3:
            continue
        
        poly_points = np.array(polygon, dtype=np.int32)
        # Fill polygon in mask - NO outline
        cv2.fillPoly(mask, [poly_points], 255, lineType=cv2.LINE_AA)
    
    return mask


def draw_room_polys_outline(
    overlay: np.ndarray,
    room_polys: List[Dict],
    color: tuple,
    thickness: int = 2,
    line_style: str = "solid",
) -> np.ndarray:
    """
    Draw outlines for room polygons.
    OUTLINES ONLY - no fill.
    Supports solid, dashed, and dotted line styles.
    """
    result = overlay.copy()
    
    for room in room_polys:
        polygon = room.get("polygon")
        if not polygon or len(polygon) < 3:
            continue
        
        poly_points = np.array(polygon, dtype=np.int32)
        
        # Draw outline based on line style
        if line_style == "solid":
            # Solid line - use standard polylines
            cv2.polylines(result, [poly_points], True, color, thickness, lineType=cv2.LINE_AA)
        elif line_style == "dashed":
            # Dashed line - draw segments
            _draw_dashed_polyline(result, poly_points, color, thickness, dash_length=12, gap_length=6)
        elif line_style == "dotted":
            # Dotted line - draw small circles along the path
            _draw_dotted_polyline(result, poly_points, color, thickness, dot_spacing=4)
        else:
            # Fallback to solid
            cv2.polylines(result, [poly_points], True, color, thickness, lineType=cv2.LINE_AA)
    
    return result


def _draw_dashed_polyline(img: np.ndarray, points: np.ndarray, color: tuple, thickness: int, dash_length: int = 12, gap_length: int = 6):
    """Draw a dashed polyline by drawing segments."""
    if len(points) < 2:
        return
    
    result = img  # Use img directly
    
    total_length = 0
    segment_lengths = []
    for i in range(len(points)):
        if i == 0:
            segment_lengths.append(0)
        else:
            p1 = points[i-1]
            p2 = points[i]
            seg_len = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            segment_lengths.append(seg_len)
            total_length += seg_len
    
    # Draw dashed segments
    current_length = 0
    dash_on = True
    remaining_dash = dash_length
    remaining_gap = 0
    
    for i in range(1, len(points)):
        p1 = points[i-1].astype(float)
        p2 = points[i].astype(float)
        seg_len = segment_lengths[i]
        
        if seg_len == 0:
            continue
        
        # Direction vector
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        pos = 0
        while pos < seg_len:
            if dash_on:
                # Draw dash
                draw_len = min(remaining_dash, seg_len - pos)
                if draw_len > 0:
                    start_pct = pos / seg_len
                    end_pct = (pos + draw_len) / seg_len
                    start_pt = (int(p1[0] + dx * start_pct), int(p1[1] + dy * start_pct))
                    end_pt = (int(p1[0] + dx * end_pct), int(p1[1] + dy * end_pct))
                    cv2.line(img, start_pt, end_pt, color, thickness, lineType=cv2.LINE_AA)
                remaining_dash -= draw_len
                if remaining_dash <= 0:
                    dash_on = False
                    remaining_gap = gap_length
                pos += draw_len
            else:
                # Skip gap
                skip_len = min(remaining_gap, seg_len - pos)
                remaining_gap -= skip_len
                if remaining_gap <= 0:
                    dash_on = True
                    remaining_dash = dash_length
                pos += skip_len
    
    # Close the polygon
    if len(points) > 2:
        p1 = points[-1]
        p2 = points[0]
        seg_len = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if seg_len > 0:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            pos = 0
            while pos < seg_len:
                if dash_on:
                    draw_len = min(remaining_dash, seg_len - pos)
                    if draw_len > 0:
                        start_pct = pos / seg_len
                        end_pct = (pos + draw_len) / seg_len
                        start_pt = (int(p1[0] + dx * start_pct), int(p1[1] + dy * start_pct))
                        end_pt = (int(p2[0] + dx * (end_pct - 1)), int(p2[1] + dy * (end_pct - 1)))
                        cv2.line(img, start_pt, end_pt, color, thickness, lineType=cv2.LINE_AA)
                    remaining_dash -= draw_len
                    if remaining_dash <= 0:
                        dash_on = False
                        remaining_gap = gap_length
                    pos += draw_len
                else:
                    skip_len = min(remaining_gap, seg_len - pos)
                    remaining_gap -= skip_len
                    if remaining_gap <= 0:
                        dash_on = True
                        remaining_dash = dash_length
                    pos += skip_len


def _draw_dotted_polyline(img: np.ndarray, points: np.ndarray, color: tuple, thickness: int, dot_spacing: int = 4):
    """Draw a dotted polyline by drawing small circles along the path."""
    if len(points) < 2:
        return
    
    # Draw dots along each segment
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]  # Wrap around for closed polygon
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        seg_len = np.sqrt(dx**2 + dy**2)
        
        if seg_len == 0:
            continue
        
        num_dots = int(seg_len / dot_spacing)
        for j in range(num_dots):
            t = j / max(num_dots, 1)
            x = int(p1[0] + dx * t)
            y = int(p1[1] + dy * t)
            cv2.circle(img, (x, y), max(1, thickness // 2), color, -1, lineType=cv2.LINE_AA)


def render_rooms_layer(
    overlay_bgr: np.ndarray,
    room_mask: Optional[np.ndarray],
    room_polys: List[Dict],
    show_fill: bool,
    show_outlines: bool,
    room_color=(255, 0, 255),  # Magenta in BGR
    outline_thickness: int = 2,
    fill_alpha: float = 0.35,
    line_style: str = "solid",  # "solid", "dashed", "dotted"
) -> np.ndarray:
    """
    Draw room fill and/or borders on overlay_bgr.
    
    - If both flags are False: draw nothing.
    - If show_fill: soft fill only (no border unless show_outlines).
    - If show_outlines: border along room edges.
    
    Args:
        overlay_bgr: BGR image to draw on
        room_mask: Optional binary mask for room regions (if available)
        room_polys: List of room dictionaries with polygon data
        show_fill: Whether to draw room fills
        show_outlines: Whether to draw room outlines
        room_color: Color for outlines (BGR tuple)
        outline_thickness: Thickness of outline in pixels
        fill_alpha: Opacity of fill (0.0 to 1.0)
        line_style: Line style for outlines ("solid", "dashed", "dotted")
    
    Returns:
        Overlay image with rooms drawn (or unchanged if both flags are False)
    """
    # HARD GUARD: Ensure show_outlines is explicitly True (not just truthy)
    # This prevents any accidental outline drawing when flag is False
    show_outlines = bool(show_outlines) and show_outlines is True
    
    if not show_fill and not show_outlines:
        # Both off - draw nothing
        logger.debug(f"[RENDER_ROOMS] Both flags False - skipping all room rendering")
        return overlay_bgr
    
    result = overlay_bgr.copy()
    
    # Draw fill if enabled - PURE FILL, NO OUTLINES
    if show_fill:
        if room_polys and len(room_polys) > 0:
            # Use polygon-based fill (distinct colors per room) - NO BORDERS
            result = draw_room_fills(result, room_polys, fill_alpha)
        elif room_mask is not None:
            # Fallback to mask-based fill if polygons not available - NO BORDERS
            result = draw_mask_fill_only(result, room_mask, room_color, fill_alpha)
    
    # Draw outlines if enabled (drawn on top of fill if both are enabled)
    # Outlines are ONLY drawn when show_outlines is EXPLICITLY True
    # HARD GUARD: Never draw outlines if flag is False
    if show_outlines is True and room_polys and len(room_polys) > 0:
        result = draw_room_polys_outline(result, room_polys, room_color, outline_thickness, line_style)
    else:
        # Explicitly skip outline drawing when flag is False
        logger.debug(f"[RENDER_ROOMS] Skipping room outlines - show_outlines={show_outlines}")
    
    return result


def draw_interior_outlines(
    image: np.ndarray,
    rooms: List[Dict],
    thickness: int = 2,
    interior_offset_px: int = 3,
) -> np.ndarray:
    """
    Draw interior outlines for rooms, offset inward from the polygon edges.
    This ensures the outline is clearly on the interior side of walls.
    """
    overlay = image.copy()
    
    outline_color = (0, 0, 255)  # Red in BGR (was white, changed to red to match user's description)
    
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.validation import make_valid
        
        for room in rooms:
            polygon = room.get("polygon")
            if not polygon or len(polygon) < 3:
                continue
            
            try:
                # Create Shapely polygon
                poly = ShapelyPolygon(polygon)
                if not poly.is_valid:
                    poly = make_valid(poly)
                
                # Offset inward to ensure outline is on interior side
                if interior_offset_px > 0:
                    offset_poly = poly.buffer(-interior_offset_px, join_style=2, cap_style=3)
                    
                    if offset_poly.is_empty:
                        # Too small after offset, use original
                        offset_poly = poly
                    else:
                        # Handle MultiPolygon - use largest
                        if hasattr(offset_poly, 'geoms'):
                            offset_poly = max(offset_poly.geoms, key=lambda p: p.area)
                    
                    # Extract coordinates
                    if offset_poly.exterior:
                        polygon = [[float(x), float(y)] for x, y in offset_poly.exterior.coords[:-1]]
            except Exception as e:
                logger.warning(f"Error offsetting room outline: {e}, using original polygon")
            
            # Draw the outline
            poly_points = np.array(polygon, dtype=np.int32)
            cv2.polylines(overlay, [poly_points], True, outline_color, thickness)
    
    except ImportError:
        # Fallback if Shapely not available - draw original polygon
        logger.warning("Shapely not available for interior offset, drawing original polygon")
        for room in rooms:
            polygon = room.get("polygon")
            if not polygon or len(polygon) < 3:
                continue
            poly_points = np.array(polygon, dtype=np.int32)
            cv2.polylines(overlay, [poly_points], True, outline_color, thickness)
    
    return overlay


def draw_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    bg_color: Optional[Tuple[int, int, int]] = None,
    font_scale: float = 0.5,
    thickness: int = 1,
):
    """Draw text label on image."""
    # HARD OFF: Skip empty text (walls will have empty text from format_label)
    if not text or text.strip() == "":
        return image
    x, y = position
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Draw background if specified
    if bg_color:
        cv2.rectangle(
            image,
            (x, y - text_height - 5),
            (x + text_width, y + baseline),
            bg_color,
            -1,
        )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_walls_from_mask(overlay: np.ndarray, wall_mask: np.ndarray, settings: Dict) -> np.ndarray:
    """
    Draw continuous walls from a processed wall mask.
    NO OUTLINES - walls are drawn as fills only (if enabled).
    
    Args:
        overlay: BGR image to draw on
        wall_mask: Binary mask where walls are 255
        settings: Dictionary with wall rendering settings
    
    Returns:
        Overlay image with walls drawn (no outlines)
    """
    # REMOVED: Wall outline drawing - no red/pink lines
    # wall_outline_thickness = settings.get("wall_box_thickness", 2)
    # wall_contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # wall_color = (255, 0, 0)  # Blue in BGR
    # cv2.drawContours(overlay, wall_contours, -1, wall_color, thickness=wall_outline_thickness)
    
    # Optionally fill walls with semi-transparent color (if enabled)
    wall_box_mode = settings.get("wall_box_mode", "filled")
    wall_box_alpha = settings.get("wall_box_alpha", 0.25)
    
    if wall_box_mode == "filled" and wall_box_alpha > 0:
        wall_color = (255, 0, 0)  # Blue in BGR
        # Create a colored overlay for walls
        wall_overlay = overlay.copy()
        wall_overlay[wall_mask > 0] = wall_color
        # Blend with original
        overlay = cv2.addWeighted(overlay, 1 - wall_box_alpha, wall_overlay, wall_box_alpha, 0)
    
    # Walls are now drawn without outlines - just the mask/fill if enabled
    return overlay


def format_label(class_name: str, confidence: float, style: str) -> str:
    """Format detection label based on style."""
    # HARD OFF: Never label walls
    if class_name == "wall":
        return ""
    if style == "simple":
        return class_name.capitalize()
    else:  # detailed
        return f"{class_name.capitalize()} {confidence:.2f}"


# LEGACY FUNCTION - DISABLED: Use unified draw_detections from utils.py instead
# This function is kept for reference but should never be called
def merge_wall_segments(wall_detections: List[Dict], merge_threshold: float = 50.0, extend_threshold: float = 100.0, img_w: int = None, img_h: int = None) -> List[Dict]:
    """
    Merge wall segments that are aligned and close together, and extend them to connect with perpendicular walls.
    
    Args:
        wall_detections: List of wall detection dicts with 'bbox' keys
        merge_threshold: Maximum distance between segments to merge (in pixels)
        extend_threshold: Maximum distance to extend walls to connect with perpendicular walls
    
    Returns:
        List of merged and extended wall detections
    """
    if not wall_detections:
        return []
    
    # Convert to list of (bbox, confidence) tuples
    walls = [(det["bbox"], det.get("confidence", 0.0)) for det in wall_detections]
    merged = []
    used = [False] * len(walls)
    
    # First pass: merge aligned segments
    for i, (bbox1, conf1) in enumerate(walls):
        if used[i]:
            continue
        
        x1, y1, x2, y2 = bbox1
        w1, h1 = abs(x2 - x1), abs(y2 - y1)
        
        # Determine if wall is horizontal or vertical
        is_horizontal = w1 > h1
        
        # Find all aligned segments to merge
        to_merge = [i]
        merged_bbox = list(bbox1)
        max_conf = conf1
        
        for j, (bbox2, conf2) in enumerate(walls):
            if i == j or used[j]:
                continue
            
            x3, y3, x4, y4 = bbox2
            w2, h2 = abs(x4 - x3), abs(y4 - y3)
            
            # Check if same orientation
            is_horizontal2 = w2 > h2
            if is_horizontal != is_horizontal2:
                continue
            
            # Calculate distance between segments
            if is_horizontal:
                # Horizontal walls: check if y-coords are close and x-coords overlap or are close
                y_center1 = (y1 + y2) / 2
                y_center2 = (y3 + y4) / 2
                y_dist = abs(y_center1 - y_center2)
                
                x_min1, x_max1 = min(x1, x2), max(x1, x2)
                x_min2, x_max2 = min(x3, x4), max(x3, x4)
                
                # Check if segments overlap or are close
                x_gap = max(0, max(x_min1, x_min2) - min(x_max1, x_max2))
                
                if y_dist <= merge_threshold and x_gap <= merge_threshold * 2:
                    to_merge.append(j)
                    merged_bbox[0] = min(merged_bbox[0], x3, x4)
                    merged_bbox[1] = min(merged_bbox[1], y3, y4)
                    merged_bbox[2] = max(merged_bbox[2], x3, x4)
                    merged_bbox[3] = max(merged_bbox[3], y3, y4)
                    max_conf = max(max_conf, conf2)
            else:
                # Vertical walls: check if x-coords are close and y-coords overlap or are close
                x_center1 = (x1 + x2) / 2
                x_center2 = (x3 + x4) / 2
                x_dist = abs(x_center1 - x_center2)
                
                y_min1, y_max1 = min(y1, y2), max(y1, y2)
                y_min2, y_max2 = min(y3, y4), max(y3, y4)
                
                # Check if segments overlap or are close
                y_gap = max(0, max(y_min1, y_min2) - min(y_max1, y_max2))
                
                if x_dist <= merge_threshold and y_gap <= merge_threshold * 2:
                    to_merge.append(j)
                    merged_bbox[0] = min(merged_bbox[0], x3, x4)
                    merged_bbox[1] = min(merged_bbox[1], y3, y4)
                    merged_bbox[2] = max(merged_bbox[2], x3, x4)
                    merged_bbox[3] = max(merged_bbox[3], y3, y4)
                    max_conf = max(max_conf, conf2)
        
        # Mark all merged segments as used
        for idx in to_merge:
            used[idx] = True
        
        # Add merged wall
        merged.append({
            "bbox": merged_bbox,
            "confidence": max_conf
        })
    
    # Second pass: extend walls to connect with perpendicular walls and find natural endpoints
    extended = []
    # Get image bounds from walls if not provided
    if img_w is None or img_h is None:
        if merged:
            all_x = []
            all_y = []
            for wall in merged:
                bbox = wall["bbox"]
                all_x.extend([bbox[0], bbox[2]])
                all_y.extend([bbox[1], bbox[3]])
            img_w = int(max(all_x)) + 500 if all_x else 5000
            img_h = int(max(all_y)) + 500 if all_y else 5000
        else:
            img_w, img_h = 5000, 5000
    
    for i, wall1 in enumerate(merged):
        bbox1 = wall1["bbox"]
        x1, y1, x2, y2 = bbox1
        w1, h1 = abs(x2 - x1), abs(y2 - y1)
        is_horizontal = w1 > h1
        
        extended_bbox = list(bbox1)
        x_min1, x_max1 = min(x1, x2), max(x1, x2)
        y_min1, y_max1 = min(y1, y2), max(y1, y2)
        
        if is_horizontal:
            # Horizontal wall: extend to meet vertical walls
            y_center1 = (y1 + y2) / 2
            left_extensions = []
            right_extensions = []
            
            for j, wall2 in enumerate(merged):
                if i == j:
                    continue
                
                bbox2 = wall2["bbox"]
                x3, y3, x4, y4 = bbox2
                w2, h2 = abs(x4 - x3), abs(y4 - y3)
                is_horizontal2 = w2 > h2
                
                # Only extend to perpendicular walls
                if is_horizontal2:
                    continue
                
                x_center2 = (x3 + x4) / 2
                y_min2, y_max2 = min(y3, y4), max(y3, y4)
                
                # Check if vertical wall intersects or is close to horizontal wall's y-level
                if y_min2 <= y_center1 <= y_max2:
                    # Check if we should extend left or right
                    if x_center2 < x_min1 and abs(x_center2 - x_min1) <= extend_threshold * 2:
                        left_extensions.append(x_center2)
                    if x_center2 > x_max1 and abs(x_center2 - x_max1) <= extend_threshold * 2:
                        right_extensions.append(x_center2)
            
            # Extend to the furthest perpendicular wall or image edge
            if left_extensions:
                extended_bbox[0] = min(extended_bbox[0], min(left_extensions))
            else:
                # Extend to image edge if no wall found
                extended_bbox[0] = max(0, extended_bbox[0] - extend_threshold)
            
            if right_extensions:
                extended_bbox[2] = max(extended_bbox[2], max(right_extensions))
            else:
                # Extend to image edge if no wall found
                extended_bbox[2] = min(img_w, extended_bbox[2] + extend_threshold)
        else:
            # Vertical wall: extend to meet horizontal walls
            x_center1 = (x1 + x2) / 2
            top_extensions = []
            bottom_extensions = []
            
            for j, wall2 in enumerate(merged):
                if i == j:
                    continue
                
                bbox2 = wall2["bbox"]
                x3, y3, x4, y4 = bbox2
                w2, h2 = abs(x4 - x3), abs(y4 - y3)
                is_horizontal2 = w2 > h2
                
                # Only extend to perpendicular walls
                if not is_horizontal2:
                    continue
                
                y_center2 = (y3 + y4) / 2
                x_min2, x_max2 = min(x3, x4), max(x3, x4)
                
                # Check if horizontal wall intersects or is close to vertical wall's x-level
                if x_min2 <= x_center1 <= x_max2:
                    # Check if we should extend up or down
                    if y_center2 < y_min1 and abs(y_center2 - y_min1) <= extend_threshold * 2:
                        top_extensions.append(y_center2)
                    if y_center2 > y_max1 and abs(y_center2 - y_max1) <= extend_threshold * 2:
                        bottom_extensions.append(y_center2)
            
            # Extend to the furthest perpendicular wall or image edge
            if top_extensions:
                extended_bbox[1] = min(extended_bbox[1], min(top_extensions))
            else:
                # Extend to image edge if no wall found
                extended_bbox[1] = max(0, extended_bbox[1] - extend_threshold)
            
            if bottom_extensions:
                extended_bbox[3] = max(extended_bbox[3], max(bottom_extensions))
            else:
                # Extend to image edge if no wall found
                extended_bbox[3] = min(img_h, extended_bbox[3] + extend_threshold)
        
        extended.append({
            "bbox": extended_bbox,
            "confidence": wall1["confidence"]
        })
    
    return extended


# LEGACY FUNCTION - DISABLED: Use unified draw_box from utils.py instead
# This function is kept for reference but should never be called
def draw_wall_line(image: np.ndarray, bbox: List[float], color: Tuple[int, int, int], thickness: int = 3) -> None:
    """
    Draw a wall as a line along its main axis (horizontal or vertical).
    This creates a continuous wall appearance rather than just a box.
    """
    x1, y1, x2, y2 = map(int, bbox)
    w, h = abs(x2 - x1), abs(y2 - y1)
    
    # Determine if wall is horizontal or vertical
    if w > h:
        # Horizontal wall - draw along the center y-coordinate
        y_center = (y1 + y2) // 2
        cv2.line(image, (x1, y_center), (x2, y_center), color, thickness)
        # Also draw a thicker rectangle for visibility
        cv2.rectangle(image, (x1, y_center - thickness), (x2, y_center + thickness), color, -1)
    else:
        # Vertical wall - draw along the center x-coordinate
        x_center = (x1 + x2) // 2
        cv2.line(image, (x_center, y1), (x_center, y2), color, thickness)
        # Also draw a thicker rectangle for visibility
        cv2.rectangle(image, (x_center - thickness, y1), (x_center + thickness, y2), color, -1)

