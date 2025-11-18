from __future__ import annotations

from dataclasses import dataclass, replace

from typing import List, Dict, Any, Tuple, Optional

import numpy as np

import cv2

import logging

from shapely.geometry import Polygon

from shapely.ops import unary_union

# Debug flag for mask dumping
DEBUG_MASKS = True

log = logging.getLogger("archimind.rooms")


@dataclass
class RoomFillSettings:
    wall_thick_px: int = 12
    door_bridge_extra_px: int = 12
    footprint_grow_px: int = 3
    min_room_area_px: int = 6000
    prefer_interior_footprint: bool = True
    outline_thickness: int = 2
    doors_as_walls: bool = True
    gap_close_px: int = 8
    scale_mode: str = "unknown"      # "unknown" | "explicit"
    explicit_scale: float = 0.0       # pixels-per-foot
    dump_masks_to: Optional[str] = None  # directory path to save barrier/free/interior PNGs


# ---------------- helpers ----------------

def _to_pixel_boxes(boxes, W, H):
    if not boxes: return []
    out = []
    for b in boxes:
        if len(b) < 4: continue
        x1,y1,x2,y2 = map(float, b[:4])
        normalized = max(abs(x1),abs(y1),abs(x2),abs(y2)) <= 1.0
        if normalized:
            # decide xywh vs xyxy
            if x2 <= 1.0 and y2 <= 1.0:
                x,y,w,h = x1*W, y1*H, x2*W, y2*H
                X1,Y1,X2,Y2 = x,y,x+w,y+h
            else:
                X1,Y1,X2,Y2 = x1*W, y1*H, x2*W, y2*H
        else:
            # pixels: guess xywh if tiny w/h
            if x2 <= 2 or y2 <= 2:
                x,y,w,h = x1,y1,x2,y2
                X1,Y1,X2,Y2 = x,y,x+w,y+h
            else:
                X1,Y1,X2,Y2 = x1,y1,x2,y2
        X1 = max(0, min(int(round(X1)), W-1))
        Y1 = max(0, min(int(round(Y1)), H-1))
        X2 = max(0, min(int(round(X2)), W-1))
        Y2 = max(0, min(int(round(Y2)), H-1))
        if X2 < X1: X1,X2 = X2,X1
        if Y2 < Y1: Y1,Y2 = Y2,Y1
        out.append([X1,Y1,X2,Y2])
    return out


def _clip_box(xyxy, w, h):
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    x1 = max(0,min(x1,w-1)); x2 = max(0,min(x2,w-1))
    y1 = max(0,min(y1,h-1)); y2 = max(0,min(y2,h-1))
    if x2 < x1: x1,x2 = x2,x1
    if y2 < y1: y1,y2 = y2,y1
    return x1,y1,x2,y2


def _draw_boxes(mask, boxes, thickness=1):
    H,W = mask.shape[:2]
    for b in boxes:
        x1,y1,x2,y2 = _clip_box(b, W, H)
        cv2.rectangle(mask, (x1,y1), (x2,y2), 255, thickness=max(1, thickness))


def build_wall_mask(walls_xyxy, image_shape, wall_thick_px=12):
    """
    Build a binary wall mask from wall detection boxes.
    
    Args:
        walls_xyxy: List of wall boxes in [x1, y1, x2, y2] format
        image_shape: (height, width) of the image
        wall_thick_px: Padding/thickness to apply to each wall box (half applied on each side)
    
    Returns:
        Binary mask (uint8) where walls are 255 and background is 0
    """
    H, W = image_shape[:2]
    wall_mask = np.zeros((H, W), dtype=np.uint8)
    
    # Calculate padding (half of wall_thick_px on each side)
    pad = max(1, wall_thick_px // 2)
    
    for box_xyxy in walls_xyxy:
        if len(box_xyxy) < 4:
            continue
        
        x1, y1, x2, y2 = map(int, box_xyxy)
        
        # Apply padding
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W - 1, x2 + pad)
        y2 = min(H - 1, y2 + pad)
        
        # Draw filled rectangle into mask
        cv2.rectangle(wall_mask, (x1, y1), (x2, y2), 255, thickness=-1)
    
    return wall_mask


def _draw_doors(mask, doors, bridge_px, as_walls):
    """
    Draw door bridges to seal door openings in the wall mask.
    
    For horizontal doors (w > h): draws a vertical bar at the center to bridge the opening.
    For vertical doors (h > w): draws a horizontal bar at the center to bridge the opening.
    
    Args:
        mask: Binary mask where walls are 255 (will be modified in-place)
        doors: List of door boxes in [x1, y1, x2, y2] format
        bridge_px: Extra pixels to extend the bridge beyond the door box
        as_walls: If True, fill entire door box; if False, draw thin bridge bar
    """
    H, W = mask.shape[:2]
    for d in doors:
        x1, y1, x2, y2 = _clip_box(d, W, H)
        w_box = x2 - x1
        h_box = y2 - y1
        
        if as_walls:
            # Fill entire door box as wall
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        else:
            # Draw thin bridge bar across door opening
            extra = bridge_px
            
            if w_box > h_box:
                # Horizontal door → bridge vertically (draw vertical bar at center)
                cx = (x1 + x2) // 2
                y1b = max(0, y1 - extra)
                y2b = min(H - 1, y2 + extra)
                cv2.rectangle(mask, (cx - 1, y1b), (cx + 1, y2b), 255, -1)
            else:
                # Vertical door → bridge horizontally (draw horizontal bar at center)
                cy = (y1 + y2) // 2
                x1b = max(0, x1 - extra)
                x2b = min(W - 1, x2 + extra)
                cv2.rectangle(mask, (x1b, cy - 1), (x2b, cy + 1), 255, -1)


def close_wall_gaps(wall_mask, gap_close_px):
    """
    Morphologically close gaps between wall boxes.
    
    Args:
        wall_mask: Binary mask where walls are 255
        gap_close_px: Size of kernel for closing operation
    
    Returns:
        Closed wall mask with gaps bridged
    """
    if gap_close_px <= 0:
        return wall_mask
    
    k = max(1, int(gap_close_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed_wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    return closed_wall_mask


def _close_gaps_oriented(barrier, gap_px):
    """
    Close gaps in the barrier mask using oriented morphological closing.
    This helps connect wall segments and close door openings.
    """
    g = max(3, int(gap_px))
    # Horizontal closing (closes vertical gaps)
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (g, 1))
    closed_h = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, k_h)
    # Vertical closing (closes horizontal gaps)
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, g))
    closed_v = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, k_v)
    # Diagonal closing (closes diagonal gaps)
    k_d1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g))
    closed_d = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, k_d1)
    # Combine all closings
    result = cv2.bitwise_or(closed_h, closed_v)
    result = cv2.bitwise_or(result, closed_d)
    return result


def _flood_kill_exterior(free_space):
    """
    Remove exterior space by flood-filling from (0,0) and borders.
    Only interior enclosed regions remain.
    
    This implements the user specification: flood-fill from (0,0) to eliminate
    the outside world, then extract contours as interior regions.
    """
    H, W = free_space.shape[:2]
    
    # Create a copy for flood-filling
    ff = free_space.copy()
    
    # Create mask for floodFill (must be 2 pixels larger on each side)
    mask = np.zeros((H + 2, W + 2), np.uint8)
    
    # Flood-fill from (0,0) to eliminate the outside world
    # This marks all connected exterior space as 0
    if ff[0, 0] == 255:  # Only flood if (0,0) is free space
        cv2.floodFill(ff, mask, (0, 0), 0)
    
    # Also flood-fill from other border points for robustness
    # (in case (0,0) is inside a room)
    border_seeds = [
        (W - 1, 0),      # Top-right
        (0, H - 1),      # Bottom-left
        (W - 1, H - 1),  # Bottom-right
    ]
    
    for seed in border_seeds:
        if 0 <= seed[0] < W and 0 <= seed[1] < H:
            if ff[seed[1], seed[0]] == 255:  # Only flood if it's free space
                mask = np.zeros((H + 2, W + 2), np.uint8)
                cv2.floodFill(ff, mask, seed, 0)
    
    # Interior is what remains as 255 (not flooded - these are enclosed regions)
    interior = (ff == 255).astype(np.uint8) * 255
    
    return interior, True


def _components(mask):
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    return [(labels==i).astype(np.uint8)*255 for i in range(1,num)]


def _poly_from_component(cm, simplify_epsilon=2.0):
    """
    Extract polygon(s) from a connected component.
    Uses simplification to smooth jagged edges.
    Returns a list of polygons (one per distinct room region).
    """
    cnts,_ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if len(c) < 3: continue
        # Simplify contour to reduce noise and smooth edges
        epsilon = simplify_epsilon * cv2.arcLength(c, True) / 100.0
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) < 3: continue
        
        pts = approx[:,0,:].astype(np.float32)
        try:
            p = Polygon(pts)
            if not p.is_valid:
                # Try to make valid
                from shapely.validation import make_valid
                p = make_valid(p)
            if p.area > 0:
                polys.append(p)
        except Exception:
            continue
    
    if not polys:
        return []
    
    # Return all valid polygons (don't merge - each is a separate room)
    return [p for p in polys if p.is_valid and p.area > 0]


def _offset_inward(poly, inset_px):
    s = poly.buffer(-max(0.0,float(inset_px)), join_style=2)
    if s.is_empty: return None
    if s.geom_type == "Polygon": return s
    if s.geom_type == "MultiPolygon": return max(list(s.geoms), key=lambda p:p.area)
    return None


def _make_room_from_contour(contour, image_shape, settings: RoomFillSettings, name_hint: str = "Room"):
    """
    Convert a contour to a room dictionary.
    
    Args:
        contour: OpenCV contour
        image_shape: (height, width) of the image
        settings: RoomFillSettings for scale calculation
        name_hint: Optional name hint for the room
    
    Returns:
        Room dictionary with polygon_wkt, centroid, area_px, area_sf, or None if invalid
    """
    if len(contour) < 3:
        return None
    
    try:
        # Simplify contour
        epsilon = 1.5 * cv2.arcLength(contour, True) / 100.0
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            return None
        
        pts = approx[:, 0, :].astype(np.float32)
        poly = Polygon(pts)
        
        if not poly.is_valid:
            from shapely.validation import make_valid
            poly = make_valid(poly)
        
        if poly.area <= 10:
            return None
        
        # Apply interior footprint offset if requested
        if settings.prefer_interior_footprint:
            shrunk = _offset_inward(poly, settings.wall_thick_px)
            if shrunk and shrunk.area > 10:
                poly = shrunk
        
        if not poly.is_valid or poly.area <= 10:
            return None
        
        cx, cy = poly.centroid.x, poly.centroid.y
        area_px = float(poly.area)
        area_sf = None
        if settings.scale_mode == "explicit" and settings.explicit_scale > 0:
            ppf = float(settings.explicit_scale)
            area_sf = area_px / (ppf * ppf)
        
        return {
            "polygon_wkt": poly.wkt,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "area_px": area_px,
            "area_sf": area_sf
        }
    except Exception as e:
        log.warning(f"Error converting contour to room: {e}")
        return None


def _run(image_shape, walls_xyxy, doors_xyxy, windows_xyxy, S: RoomFillSettings):
    H,W = image_shape[:2]
    dbg: Dict[str,Any] = {}

    # Step 1: Build wall mask from boxes with padding
    wall_mask = build_wall_mask(walls_xyxy, image_shape, S.wall_thick_px)
    wall_pixels = np.count_nonzero(wall_mask)
    log.info(f"wall_mask: shape={wall_mask.shape}, wall_pixels={wall_pixels}, total_pixels={H*W}")
    
    # Step 2: Morphologically close gaps between wall boxes
    closed_wall_mask = close_wall_gaps(wall_mask, S.gap_close_px)
    closed_wall_pixels = np.count_nonzero(closed_wall_mask)
    log.info(f"closed_mask: wall_pixels={closed_wall_pixels} (increased by {closed_wall_pixels - wall_pixels})")
    
    # Verify mask polarity: closed_mask should have 255 for walls, 0 for free space
    if closed_wall_pixels == 0:
        log.warning("closed_mask has no wall pixels - check wall detections!")
    
    # Step 3: Add doors to the barrier mask
    barrier = closed_wall_mask.copy()
    _draw_doors(barrier, doors_xyxy, S.door_bridge_extra_px, S.doors_as_walls)
    barrier_pixels = np.count_nonzero(barrier)
    log.info(f"barrier (after doors): wall_pixels={barrier_pixels}")
    
    # Additional small dilation to ensure walls and doors are fully connected
    if S.gap_close_px > 0:
        k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        barrier = cv2.dilate(barrier, k_small, iterations=1)
        barrier_pixels_after_dilate = np.count_nonzero(barrier)
        log.info(f"barrier (after dilation): wall_pixels={barrier_pixels_after_dilate}")

    # Invert barrier to get free space: 255 = free, 0 = wall
    free_space = cv2.bitwise_not(barrier)
    free_space_pixels = np.count_nonzero(free_space)
    log.info(f"free_space (inverted barrier): free_pixels={free_space_pixels}")
    
    # Flood-fill to remove exterior
    interior, flood_ok = _flood_kill_exterior(free_space)
    interior_pixels = np.count_nonzero(interior)
    log.info(f"ff (filled interior): nonzero={interior_pixels}, flood_ok={flood_ok}")
    dbg["border_flood_ok"] = bool(flood_ok)
    
    # Log parameters
    log.info(f"Parameters: min_room_area_px={S.min_room_area_px}, gap_close_px={S.gap_close_px}, "
             f"wall_thick_px={S.wall_thick_px}, door_bridge_extra_px={S.door_bridge_extra_px}, "
             f"doors_as_walls={S.doors_as_walls}")

    # Debug mask dumping
    if DEBUG_MASKS or S.dump_masks_to:
        from pathlib import Path
        if S.dump_masks_to:
            dump_path = Path(S.dump_masks_to)
        else:
            dump_path = Path("debug_masks")
        dump_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dump_path / "1_wall_mask.png"), wall_mask)
        cv2.imwrite(str(dump_path / "2_closed_mask.png"), closed_wall_mask)
        cv2.imwrite(str(dump_path / "3_barrier.png"), barrier)
        cv2.imwrite(str(dump_path / "4_free_space.png"), free_space)
        cv2.imwrite(str(dump_path / "5_ff_interior.png"), interior)
        log.info(f"Debug masks saved to {dump_path}")

    if not np.any(interior):
        log.warning("No interior regions found after flood-fill - all space is exterior")
        return [], {**dbg,
            "interior_components_pre_filter":0,
            "interior_components_post_filter":0,
            "skipped_reason":"No enclosed regions found — increase wall_thick_px or gap_close_px."}

    # Step 5: Clean up interior mask (remove small holes and smooth edges)
    if S.footprint_grow_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (S.footprint_grow_px, S.footprint_grow_px))
        # Close small holes
        interior = cv2.morphologyEx(interior, cv2.MORPH_CLOSE, k)
        # Smooth edges
        interior = cv2.dilate(interior, k, iterations=1)
        interior = cv2.erode(interior, k, iterations=1)
        interior_pixels_after_cleanup = np.count_nonzero(interior)
        log.info(f"interior (after cleanup): nonzero={interior_pixels_after_cleanup}")

    # Find contours on the interior mask (ff)
    # Verify we're using the correct mask: interior should have 255 for rooms, 0 for walls/exterior
    contours, hierarchy = cv2.findContours(interior, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    log.info(f"Contours found: {len(contours)}")
    
    # Use effective minimum area (clamped to at least 200px)
    effective_min_area = max(200, S.min_room_area_px)
    log.info(f"Using effective_min_area={effective_min_area} (requested={S.min_room_area_px})")
    
    # Filter contours by area
    candidate_rooms = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area >= effective_min_area:
            candidate_rooms.append(cnt)
            log.debug(f"Contour {i}: area={area:.1f} px² (PASS)")
        else:
            log.debug(f"Contour {i}: area={area:.1f} px² (FAIL - below {effective_min_area})")
    
    log.info(f"Candidate rooms after area filter: {len(candidate_rooms)}")
    
    # Convert candidate contours to components for processing
    comps = _components(interior)
    dbg["interior_components_pre_filter"] = len(comps)
    
    # Filter components by area using effective_min_area
    kept = []
    for cm in comps:
        area = int(np.count_nonzero(cm))
        if area >= effective_min_area:
            kept.append(cm)
    
    dbg["interior_components_post_filter"] = len(kept)
    
    if not kept and not candidate_rooms:
        log.warning("Room detection: no rooms after filtering; consider leak or thresholds too high")
        return [], {**dbg, "skipped_reason":"All interior regions filtered — lower min_room_area_px or increase footprint_grow_px."}

    rooms = []
    for cm in kept:
        # Extract all polygons from this component (may be multiple separate rooms)
        polys = _poly_from_component(cm, simplify_epsilon=1.5)
        
        for poly in polys:
            if poly is None: continue
            
            # Apply interior footprint offset if requested
            if S.prefer_interior_footprint:
                shrunk = _offset_inward(poly, S.wall_thick_px)
                if shrunk and shrunk.area > 10:
                    poly = shrunk
            
            # Validate polygon
            if (not poly.is_valid) or poly.area <= 10:
                continue
            
            # Handle MultiPolygon - extract each sub-polygon as a separate room
            if hasattr(poly, 'geoms'):
                # It's a MultiPolygon - process each sub-polygon separately
                for sub_poly in poly.geoms:
                    if sub_poly.is_valid and sub_poly.area > 10:
                        cx, cy = sub_poly.centroid.x, sub_poly.centroid.y
                        area_px = float(sub_poly.area)
                        area_sf = None
                        if S.scale_mode == "explicit" and S.explicit_scale > 0:
                            ppf = float(S.explicit_scale)
                            area_sf = area_px / (ppf * ppf)
                        
                        rooms.append({
                            "polygon_wkt": sub_poly.wkt,
                            "centroid_x": float(cx),
                            "centroid_y": float(cy),
                            "area_px": area_px,
                            "area_sf": area_sf
                        })
            else:
                # Single polygon
                cx, cy = poly.centroid.x, poly.centroid.y
                area_px = float(poly.area)
                area_sf = None
                if S.scale_mode == "explicit" and S.explicit_scale > 0:
                    ppf = float(S.explicit_scale)
                    area_sf = area_px / (ppf * ppf)
                
                rooms.append({
                    "polygon_wkt": poly.wkt,
                    "centroid_x": float(cx),
                    "centroid_y": float(cy),
                    "area_px": area_px,
                    "area_sf": area_sf
                })

    log.info(f"Rooms extracted from components: {len(rooms)}")
    
    # Fallback: if no rooms found, use largest contour as footprint room
    if not rooms:
        log.warning("No rooms found after normal pipeline - attempting fallback: using largest interior contour")
        contours2, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours2:
            largest = max(contours2, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest)
            log.info(f"Largest contour area: {largest_area:.1f} px²")
            
            if largest_area >= 200:  # Only use if reasonably large
                footprint_room = _make_room_from_contour(largest, image_shape, S, name_hint="Footprint")
                if footprint_room:
                    rooms.append(footprint_room)
                    log.warning("Room detection fallback: using largest interior contour as single footprint room")
                    dbg["fallback_used"] = True
                else:
                    log.warning("Failed to convert largest contour to room polygon")
            else:
                log.warning(f"Largest contour too small ({largest_area:.1f} px²) - skipping fallback")
        else:
            log.warning("No contours found for fallback")

    dbg["room_polygon_count"] = len(rooms)
    log.info(f"Final room count: {len(rooms)}")
    return rooms, dbg, closed_wall_mask


def infer_rooms_from_detections(image_shape, walls_xyxy, doors_xyxy, windows_xyxy, settings: RoomFillSettings):
    H,W = image_shape[:2]

    walls_xyxy   = _to_pixel_boxes(walls_xyxy,   W, H)
    doors_xyxy   = _to_pixel_boxes(doors_xyxy,   W, H)
    windows_xyxy = _to_pixel_boxes(windows_xyxy, W, H)

    rooms, dbg, closed_wall_mask = _run(image_shape, walls_xyxy, doors_xyxy, windows_xyxy, settings)
    if rooms:
        return {"rooms": rooms, "debug": dbg, "closed_wall_mask": closed_wall_mask}

    # Rescue pass
    strong = replace(
        settings,
        wall_thick_px=max(settings.wall_thick_px, 14),
        door_bridge_extra_px=max(settings.door_bridge_extra_px, 14),
        gap_close_px=max(settings.gap_close_px, 28),
        footprint_grow_px=max(settings.footprint_grow_px, 4),
        min_room_area_px=max(1000, int(settings.min_room_area_px*0.5)),
        doors_as_walls=True
    )
    rooms2, dbg2, closed_wall_mask2 = _run(image_shape, walls_xyxy, doors_xyxy, windows_xyxy, strong)
    if rooms2:
        dbg2["retry_used"] = True
        return {"rooms": rooms2, "debug": dbg2, "closed_wall_mask": closed_wall_mask2}
    dbg2["skipped_reason"] = dbg2.get("skipped_reason") or "Polygonization produced no valid rooms."
    return {"rooms": [], "debug": dbg2, "skipped_reason": dbg2["skipped_reason"], "closed_wall_mask": closed_wall_mask2}
