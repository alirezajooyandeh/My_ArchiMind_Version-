"""Geometry operations for room polygonization and area calculations."""
import logging
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.validation import make_valid

logger = logging.getLogger(__name__)


def build_room_polygons(
    detections: Dict[str, List[Dict]],
    image_shape: Tuple[int, int],
    door_bridge_extra_px: int = 6,
    min_room_area_px: int = 1000,
    footprint_grow_px: int = 2,
    prefer_interior_footprint: bool = True,
    wall_thick_px: int = 8,
) -> List[Dict]:
    """
    Build room polygons from wall, door, and room detections.
    
    Args:
        detections: Dictionary with 'wall', 'door', 'window', 'room' keys
        image_shape: (height, width) of the image
        door_bridge_extra_px: Extra pixels to bridge door gaps (not used - doors are boundaries)
        min_room_area_px: Minimum area to keep a room
        footprint_grow_px: Morphology grow size for room footprints
        prefer_interior_footprint: Clip to interior walls
        wall_thick_px: Wall thickness hint for interior clipping
    
    Returns:
        List of room dictionaries with polygon, area, centroid, etc.
    """
    h, w = image_shape[:2]
    
    # Create binary masks for walls and doors
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    door_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw walls - make them slightly thicker to ensure separation
    for wall in detections.get("wall", []):
        bbox = wall["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        # Expand walls slightly to ensure proper room separation
        x1 = max(0, x1 - 2)
        y1 = max(0, y1 - 2)
        x2 = min(w, x2 + 2)
        y2 = min(h, y2 + 2)
        cv2.rectangle(wall_mask, (x1, y1), (x2, y2), 255, -1)
    
    # Draw doors as boundaries (not openings) - treat them like walls
    for door in detections.get("door", []):
        bbox = door["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        # Doors are boundaries, so draw them as solid barriers
        # Expand slightly to ensure proper separation
        x1 = max(0, x1 - 2)
        y1 = max(0, y1 - 2)
        x2 = min(w, x2 + 2)
        y2 = min(h, y2 + 2)
        cv2.rectangle(door_mask, (x1, y1), (x2, y2), 255, -1)
    
    # Combine wall and door masks - both act as barriers
    barrier_mask = cv2.bitwise_or(wall_mask, door_mask)
    
    # Dilate barriers slightly to ensure proper room separation
    kernel_dilate = np.ones((5, 5), np.uint8)
    barrier_mask = cv2.dilate(barrier_mask, kernel_dilate, iterations=1)
    
    # If we have room detections with polygons, use them
    rooms = []
    room_detections = detections.get("room", [])
    total_image_area = h * w
    max_room_area_ratio = 0.5  # A single room shouldn't cover more than 50% of the image
    
    if room_detections:
        # First, filter out overly large room detections (likely false positives)
        filtered_room_dets = []
        for room_det in room_detections:
            bbox = room_det.get("bbox", [])
            if bbox:
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if bbox_area / total_image_area > max_room_area_ratio:
                    logger.warning(f"Skipping room detection covering {bbox_area/total_image_area*100:.1f}% of image")
                    continue
            filtered_room_dets.append(room_det)
        
        # If we have valid room detections, process them
        if filtered_room_dets:
            for i, room_det in enumerate(filtered_room_dets):
                polygon = room_det.get("polygon")
                bbox = room_det.get("bbox")
                
                # If we have a polygon, use it; otherwise use bbox as starting point
                if polygon and len(polygon) >= 3:
                    # Convert polygon to mask
                    room_mask = np.zeros((h, w), dtype=np.uint8)
                    poly_points = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(room_mask, [poly_points], 255)
                elif bbox:
                    # Use bbox as starting point
                    x1, y1, x2, y2 = map(int, bbox)
                    room_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(room_mask, (x1, y1), (x2, y2), 255, -1)
                else:
                    continue
                
                # Remove barriers (walls and doors) from room
                room_mask = cv2.bitwise_and(room_mask, cv2.bitwise_not(barrier_mask))
                
                # Morphology operations to clean up and separate rooms
                kernel = np.ones((footprint_grow_px * 2 + 1, footprint_grow_px * 2 + 1), np.uint8)
                room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, kernel)
                room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, kernel)
                
                # Find all separate room regions using connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(room_mask, connectivity=8)
                
                for label_id in range(1, num_labels):  # Skip background (label 0)
                    area_px = stats[label_id, cv2.CC_STAT_AREA]
                    if area_px < min_room_area_px:
                        continue
                    
                    # Check if this room is too large (shouldn't cover more than max_room_area_ratio)
                    if area_px / total_image_area > max_room_area_ratio:
                        logger.warning(f"Skipping room component covering {area_px/total_image_area*100:.1f}% of image")
                        continue
                    
                    # Extract mask for this component
                    component_mask = (labels == label_id).astype(np.uint8) * 255
                    
                    # Find contour
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    
                    # Use largest contour
                    contour = max(contours, key=cv2.contourArea)
                    
                    # Simplify contour to polygon
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) < 3:
                        continue
                    
                    polygon_points = [[float(p[0][0]), float(p[0][1])] for p in approx]
                    
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])
                    else:
                        # Fallback to polygon centroid
                        poly = Polygon(polygon_points)
                        if poly.is_valid:
                            cx, cy = poly.centroid.x, poly.centroid.y
                        else:
                            continue
                    
                    rooms.append({
                        "id": len(rooms),
                        "polygon": polygon_points,
                        "area_px": float(area_px),
                        "centroid": [cx, cy],
                        "label": f"Room {len(rooms) + 1}",
                    })
    
    # Always use wall-based inference for floor plans - it's more reliable
    # Room detections from YOLO are often inaccurate for floor plans
    logger.info(f"Got {len(rooms)} rooms from detections, inferring from wall/door layout")
    inferred_rooms = infer_rooms_from_walls(
        wall_mask, door_mask, h, w,
        min_room_area_px, footprint_grow_px
    )
    
    # Use inferred rooms if we got better results, or if we got too few from detections
    if len(inferred_rooms) >= len(rooms) and len(inferred_rooms) >= 2:
        rooms = inferred_rooms
        logger.info(f"Using {len(rooms)} rooms inferred from wall/door layout")
    elif len(rooms) < 2:
        rooms = inferred_rooms
        logger.info(f"Only {len(rooms)} rooms from detections, using {len(inferred_rooms)} inferred rooms")
    
    # Apply interior footprint clipping if requested
    if prefer_interior_footprint and rooms:
        rooms = clip_to_interior(rooms, wall_mask, wall_thick_px)
    
    return rooms


def infer_rooms_from_walls(
    wall_mask: np.ndarray,
    door_mask: np.ndarray,
    h: int,
    w: int,
    min_room_area_px: int,
    footprint_grow_px: int,
) -> List[Dict]:
    """Infer room polygons from wall and door layout using connected components."""
    # Combine barriers - both walls and doors act as boundaries
    barrier_mask = cv2.bitwise_or(wall_mask, door_mask)
    
    # Dilate barriers to ensure proper room separation
    # This ensures rooms are properly separated by walls and doors
    kernel_dilate = np.ones((5, 5), np.uint8)
    barrier_mask = cv2.dilate(barrier_mask, kernel_dilate, iterations=2)
    
    # Invert to get potential room areas (interior spaces)
    room_areas = cv2.bitwise_not(barrier_mask)
    
    # Morphology to clean up small gaps and separate rooms
    kernel = np.ones((max(3, footprint_grow_px * 2 + 1), max(3, footprint_grow_px * 2 + 1)), np.uint8)
    # Close small gaps
    room_areas = cv2.morphologyEx(room_areas, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Remove small noise
    room_areas = cv2.morphologyEx(room_areas, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find connected components - each component is a separate room
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(room_areas, connectivity=8)
    
    rooms = []
    total_image_area = h * w
    max_room_area_ratio = 0.4  # A single room shouldn't cover more than 40% of the image
    min_room_area_ratio = 0.001  # Minimum 0.1% of image
    
    # Sort by area (largest first) to prioritize significant rooms
    room_candidates = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area_px = stats[i, cv2.CC_STAT_AREA]
        area_ratio = area_px / total_image_area
        
        # Filter by area
        if area_px < min_room_area_px:
            continue
        if area_ratio < min_room_area_ratio:
            continue
        if area_ratio > max_room_area_ratio:
            logger.warning(f"Skipping room component covering {area_ratio*100:.1f}% of image (too large)")
            continue
        
        room_candidates.append((i, area_px, area_ratio))
    
    # Sort by area (largest first)
    room_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Process each room candidate
    for label_id, area_px, area_ratio in room_candidates:
        # Extract mask for this component
        component_mask = (labels == label_id).astype(np.uint8) * 255
        
        # Find contour
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Simplify to polygon with appropriate epsilon
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            continue
        
        polygon_points = [[float(p[0][0]), float(p[0][1])] for p in approx]
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx, cy = float(centroids[label_id][0]), float(centroids[label_id][1])
        
        rooms.append({
            "id": len(rooms),
            "polygon": polygon_points,
            "area_px": float(area_px),
            "centroid": [cx, cy],
            "label": f"Room {len(rooms) + 1}",
        })
    
    logger.info(f"Inferred {len(rooms)} rooms from wall/door layout")
    return rooms


def clip_to_interior(
    rooms: List[Dict],
    wall_mask: np.ndarray,
    wall_thick_px: int,
) -> List[Dict]:
    """Clip room polygons to interior by offsetting inward."""
    try:
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import transform
        import math
        
        clipped_rooms = []
        
        for room in rooms:
            polygon_points = room["polygon"]
            if len(polygon_points) < 3:
                continue
            
            try:
                poly = ShapelyPolygon(polygon_points)
                if not poly.is_valid:
                    poly = make_valid(poly)
                
                # Offset inward
                offset_poly = poly.buffer(-wall_thick_px, join_style=2, cap_style=3)
                
                if offset_poly.is_empty:
                    # Too small after offset, skip
                    continue
                
                # Handle MultiPolygon
                if hasattr(offset_poly, 'geoms'):
                    # Use largest polygon
                    offset_poly = max(offset_poly.geoms, key=lambda p: p.area)
                
                # Convert back to list
                if offset_poly.exterior:
                    clipped_points = [[float(x), float(y)] for x, y in offset_poly.exterior.coords[:-1]]
                    
                    if len(clipped_points) >= 3:
                        # Recalculate area
                        clipped_poly = ShapelyPolygon(clipped_points)
                        area_px = clipped_poly.area
                        
                        room_copy = room.copy()
                        room_copy["polygon"] = clipped_points
                        room_copy["area_px"] = float(area_px)
                        room_copy["centroid"] = [float(clipped_poly.centroid.x), float(clipped_poly.centroid.y)]
                        clipped_rooms.append(room_copy)
                
            except Exception as e:
                logger.warning(f"Error clipping room {room.get('id')}: {e}, using original")
                clipped_rooms.append(room)
        
        return clipped_rooms if clipped_rooms else rooms
        
    except Exception as e:
        logger.warning(f"Error in clip_to_interior: {e}, returning original rooms")
        return rooms


def calculate_area_sf(area_px: float, pixels_per_foot: float) -> Optional[float]:
    """Convert pixel area to square feet."""
    if pixels_per_foot <= 0:
        return None
    
    # Area in square feet = (area in pixels) / (pixels_per_foot^2)
    area_sf = area_px / (pixels_per_foot ** 2)
    return round(area_sf, 2)


def polygon_to_wkt(polygon: List[List[float]]) -> str:
    """Convert polygon coordinates to WKT format."""
    if not polygon or len(polygon) < 3:
        return ""
    
    # Close the polygon if not closed
    coords = polygon.copy()
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    
    coord_str = ", ".join([f"{x} {y}" for x, y in coords])
    return f"POLYGON(({coord_str}))"

