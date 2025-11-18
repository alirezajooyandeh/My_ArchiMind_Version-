"""Improved room polygonization with exterior removal and gap tolerance."""
import logging
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from shapely.geometry import Polygon
from shapely.validation import make_valid

logger = logging.getLogger(__name__)


def build_room_polygons_v2(
    detections: Dict[str, List[Dict]],
    image_shape: Tuple[int, int],
    wall_conf: float = 0.60,
    door_conf: float = 0.50,
    wind_conf: float = 0.80,
    door_bridge_extra_px: int = 12,
    wall_thick_px: int = 10,
    min_room_area_px: int = 6000,
    footprint_grow_px: int = 3,
    prefer_interior_footprint: bool = True,
) -> Tuple[List[Dict], List[str], Dict]:
    """
    Build room polygons using the specified algorithm.
    
    Returns:
        Tuple of (rooms_list, warnings_list, diagnostics_dict)
    """
    h, w = image_shape[:2]
    warnings = []
    diagnostics = {
        "models_loaded": {
            "wall": "wall" in detections,
            "door": "door" in detections,
            "window": "window" in detections,
        },
        "counts": {
            "walls": len([d for d in detections.get("wall", []) if d.get("confidence", 0) >= wall_conf]),
            "doors": len([d for d in detections.get("door", []) if d.get("confidence", 0) >= door_conf]),
            "windows": len([d for d in detections.get("window", []) if d.get("confidence", 0) >= wind_conf]),
        },
        "border_flood_ok": False,
        "interior_components_pre_filter": 0,
        "interior_components_post_filter": 0,
        "room_polygon_count": 0,
        "skipped_reason": None,
    }
    
    # Step 1: Barrier mask
    barrier_mask, barrier_warnings, barrier_diagnostics = create_barrier_mask(
        detections, h, w, wall_conf, door_conf, door_bridge_extra_px, wall_thick_px
    )
    warnings.extend(barrier_warnings)
    diagnostics.update(barrier_diagnostics)
    
    if barrier_mask is None:
        diagnostics["skipped_reason"] = "Missing wall detections — cannot form closed rooms."
        return [], warnings, diagnostics
    
    # Step 2: Free space (invert barrier)
    free_space = cv2.bitwise_not(barrier_mask)
    
    # Step 3: Exterior kill (mandatory)
    interior_mask, exterior_warnings, exterior_diagnostics = remove_exterior(free_space, h, w)
    warnings.extend(exterior_warnings)
    diagnostics.update(exterior_diagnostics)
    
    # Check if interior mask is empty (no enclosed regions)
    if interior_mask is None or np.sum(interior_mask > 0) == 0:
        diagnostics["skipped_reason"] = "Exterior not removed — increase wall_thick_px or door_bridge_extra_px."
        diagnostics["border_flood_ok"] = False
        return [], warnings, diagnostics
    
    # Step 4: Component extraction
    rooms, component_warnings, component_diagnostics = extract_room_components(
        interior_mask, min_room_area_px, footprint_grow_px
    )
    warnings.extend(component_warnings)
    diagnostics.update(component_diagnostics)
    
    if not rooms:
        if diagnostics.get("interior_components_pre_filter", 0) == 0:
            diagnostics["skipped_reason"] = "No enclosed regions found — raise imgsz or lower thresholds."
        else:
            diagnostics["skipped_reason"] = "All interiors filtered — lower min_room_area_px or increase footprint_grow_px."
        return [], warnings, diagnostics
    
    # Step 5: Polygonization and cleanup
    rooms = polygonize_rooms(rooms, prefer_interior_footprint, wall_thick_px)
    diagnostics["room_polygon_count"] = len(rooms)
    
    return rooms, warnings, diagnostics


def create_barrier_mask(
    detections: Dict[str, List[Dict]],
    h: int,
    w: int,
    wall_conf: float,
    door_conf: float,
    door_bridge_extra_px: int,
    wall_thick_px: int,
) -> Tuple[Optional[np.ndarray], List[str], Dict]:
    """Step 1: Create barrier mask from walls and doors."""
    warnings = []
    diagnostics = {}
    
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    door_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Rasterize wall boxes using wall_conf
    wall_detections = [d for d in detections.get("wall", []) if d.get("confidence", 0) >= wall_conf]
    
    if not wall_detections:
        warnings.append("Missing walls: cannot fill rooms. Increase wall_conf or check wall weights.")
        return None, warnings, diagnostics
    
    for wall in wall_detections:
        bbox = wall["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        # Ensure valid coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(wall_mask, (x1, y1), (x2, y2), 255, -1)
    
    # Seal doors: draw bridges across door openings (perpendicular to wall direction)
    door_detections = [d for d in detections.get("door", []) if d.get("confidence", 0) >= door_conf]
    
    for door in door_detections:
        bbox = door["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            # Determine door orientation (horizontal or vertical)
            width = x2 - x1
            height = y2 - y1
            
            if width > height:
                # Horizontal door - bridge vertically (perpendicular)
                bridge_y = (y1 + y2) // 2
                bridge_y1 = max(0, bridge_y - door_bridge_extra_px // 2)
                bridge_y2 = min(h, bridge_y + door_bridge_extra_px // 2)
                cv2.rectangle(door_mask, (x1, bridge_y1), (x2, bridge_y2), 255, -1)
            else:
                # Vertical door - bridge horizontally (perpendicular)
                bridge_x = (x1 + x2) // 2
                bridge_x1 = max(0, bridge_x - door_bridge_extra_px // 2)
                bridge_x2 = min(w, bridge_x + door_bridge_extra_px // 2)
                cv2.rectangle(door_mask, (bridge_x1, y1), (bridge_x2, y2), 255, -1)
    
    # Combine walls and doors (windows are NOT sealed)
    barrier_mask = cv2.bitwise_or(wall_mask, door_mask)
    
    # Dilate to close micro-gaps
    kernel = np.ones((wall_thick_px, wall_thick_px), np.uint8)
    barrier_mask = cv2.dilate(barrier_mask, kernel, iterations=1)
    
    diagnostics["wall_mask_coverage"] = float(np.sum(wall_mask > 0) / (h * w))
    diagnostics["door_mask_coverage"] = float(np.sum(door_mask > 0) / (h * w))
    diagnostics["barrier_mask_coverage"] = float(np.sum(barrier_mask > 0) / (h * w))
    
    return barrier_mask, warnings, diagnostics


def remove_exterior(free_space: np.ndarray, h: int, w: int):
    """
    Given a free_space mask (nonzero = free), remove everything that is connected to the outside.
    Returns:
      interior_mask  : uint8 {0,255} free-space pixels that are fully enclosed by walls
      warnings       : list[str]
      diagnostics    : dict
    Notes:
      - Ignores the passed h,w and derives shape from free_space to avoid size mismatches.
      - Uses 1px padding on the working image and builds a mask of shape (H+2, W+2) as required by OpenCV.
      - Uses FLOODFILL_MASK_ONLY so the image itself is not mutated.
    """
    warnings = []
    diagnostics = {}

    # 0) Normalize input to single-channel uint8 {0,255}
    img0 = free_space
    if img0.ndim == 3:
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img0 = (img0 > 0).astype(np.uint8) * 255

    H0, W0 = img0.shape[:2]

    # 1) Pad the free-space image by 1px black border (treated as NOT free)
    padded = cv2.copyMakeBorder(img0, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)

    # 2) Create mask with shape (H+2, W+2) where H,W are padded image dims (OpenCV requirement)
    H, W = padded.shape[:2]
    mask = np.zeros((H + 2, W + 2), dtype=np.uint8)

    # 3) Flood-fill from any free pixel on the *inner* border of the padded image.
    #    We don't touch the artificial outermost padded border (indices 0 and H-1 / W-1),
    #    we scan the ring just inside it (rows 1..H-2, cols 1..W-2).
    flags = cv2.FLOODFILL_MASK_ONLY | 4 | (255 << 8)  # connectivity-4; write 255s into mask

    # top and bottom scan lines inside the padding
    y_top, y_bot = 1, H - 2
    for x in range(1, W - 1):
        if padded[y_top, x] == 255:
            cv2.floodFill(padded, mask, (x, y_top), 0, flags=flags)
        if padded[y_bot, x] == 255:
            cv2.floodFill(padded, mask, (x, y_bot), 0, flags=flags)

    # left and right scan columns inside the padding
    x_left, x_right = 1, W - 2
    for y in range(1, H - 1):
        if padded[y, x_left] == 255:
            cv2.floodFill(padded, mask, (x_left, y), 0, flags=flags)
        if padded[y, x_right] == 255:
            cv2.floodFill(padded, mask, (x_right, y), 0, flags=flags)

    # 4) mask is (H+2, W+2). The filled region (value > 0) marks exterior-connected free space.
    exterior_mask_padded = (mask[1:H+1, 1:W+1] > 0).astype(np.uint8) * 255  # strip mask's 1px rim

    # 5) Interior free-space = free minus exterior; then remove the 1px padding we added
    free_padded = (padded > 0).astype(np.uint8) * 255
    interior_padded = cv2.bitwise_and(free_padded, cv2.bitwise_not(exterior_mask_padded))
    interior = interior_padded[1:H-1, 1:W-1]  # back to original size

    # 6) Sanity checks / diagnostics
    exterior_area = int(np.count_nonzero(exterior_mask_padded))
    interior_area = int(np.count_nonzero(interior))
    free_area     = int(np.count_nonzero(img0))

    diagnostics.update({
        "H0": H0, "W0": W0,
        "H_padded": H, "W_padded": W,
        "free_area": free_area,
        "exterior_area": exterior_area,
        "interior_area": interior_area,
        "border_flood_ok": interior_area > 0,
    })

    if interior_area == 0 and free_area > 0:
        warnings.append("All free space appears connected to the exterior (no enclosed rooms).")
    if exterior_area == 0 and free_area > 0:
        warnings.append("No exterior-connected free space found; image may be fully enclosed or over-sealed.")

    return interior.astype(np.uint8), warnings, diagnostics


def extract_room_components(
    interior_mask: np.ndarray,
    min_room_area_px: int,
    footprint_grow_px: int,
) -> Tuple[List[Dict], List[str], Dict]:
    """Step 4: Extract room components using morphology and connected components."""
    warnings = []
    diagnostics = {}
    
    # Small grow-then-shrink to close pinholes
    if footprint_grow_px > 0:
        kernel = np.ones((footprint_grow_px * 2 + 1, footprint_grow_px * 2 + 1), np.uint8)
        interior_mask = cv2.morphologyEx(interior_mask, cv2.MORPH_CLOSE, kernel)
        interior_mask = cv2.morphologyEx(interior_mask, cv2.MORPH_OPEN, kernel)
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(interior_mask, connectivity=8)
    
    diagnostics["interior_components_pre_filter"] = num_labels - 1  # Exclude background
    
    rooms = []
    filtered_count = 0
    for i in range(1, num_labels):  # Skip background (label 0)
        area_px = stats[i, cv2.CC_STAT_AREA]
        if area_px < min_room_area_px:
            filtered_count += 1
            continue
        
        # Extract component mask
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Find contour
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            filtered_count += 1
            continue
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Store for polygonization
        rooms.append({
            "id": len(rooms),
            "contour": contour,
            "area_px": float(area_px),
            "centroid": [float(centroids[i][0]), float(centroids[i][1])],
        })
    
    diagnostics["interior_components_post_filter"] = len(rooms)
    diagnostics["components_filtered"] = filtered_count
    
    if not rooms:
        if diagnostics["interior_components_pre_filter"] == 0:
            warnings.append("No enclosed regions found — raise imgsz or lower thresholds.")
        else:
            warnings.append("All interiors filtered — lower min_room_area_px or increase footprint_grow_px.")
    
    return rooms, warnings, diagnostics


def polygonize_rooms(
    rooms: List[Dict],
    prefer_interior_footprint: bool,
    wall_thick_px: int,
) -> List[Dict]:
    """Step 5: Convert contours to polygons and optionally offset inward."""
    result_rooms = []
    
    for room in rooms:
        contour = room["contour"]
        
        # Simplify polygon
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            continue
        
        polygon_points = [[float(p[0][0]), float(p[0][1])] for p in approx]
        
        # Optionally offset inward
        if prefer_interior_footprint:
            try:
                poly = Polygon(polygon_points)
                if not poly.is_valid:
                    poly = make_valid(poly)
                
                # Offset inward
                offset_poly = poly.buffer(-wall_thick_px, join_style=2, cap_style=3)
                
                if offset_poly.is_empty:
                    # Too small after offset, use original
                    pass
                else:
                    # Handle MultiPolygon
                    if hasattr(offset_poly, 'geoms'):
                        offset_poly = max(offset_poly.geoms, key=lambda p: p.area)
                    
                    if offset_poly.exterior:
                        polygon_points = [[float(x), float(y)] for x, y in offset_poly.exterior.coords[:-1]]
                        # Recalculate area
                        poly = Polygon(polygon_points)
                        room["area_px"] = float(poly.area)
                        room["centroid"] = [float(poly.centroid.x), float(poly.centroid.y)]
            except Exception as e:
                logger.warning(f"Error offsetting room {room['id']}: {e}, using original polygon")
        
        result_rooms.append({
            "id": room["id"],
            "polygon": polygon_points,
            "area_px": room["area_px"],
            "centroid": room["centroid"],
            "label": f"Room {room['id'] + 1}",
        })
    
    return result_rooms

