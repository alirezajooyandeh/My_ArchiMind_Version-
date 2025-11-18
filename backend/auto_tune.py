"""Auto-tune logic for optimizing detection parameters."""
import logging
import numpy as np
from typing import Dict, List, Tuple
import time

logger = logging.getLogger(__name__)


def auto_tune_parameters(
    model_manager,
    image: np.ndarray,
    candidate_imgsz: List[int] = None,
    base_conf_thresholds: Dict[str, float] = None,
) -> Dict:
    """
    Auto-tune imgsz and confidence thresholds based on plan characteristics.
    
    Returns:
        Dictionary with optimized imgsz and conf_thresholds
    """
    if candidate_imgsz is None:
        candidate_imgsz = [896, 1152, 1280]
    
    if base_conf_thresholds is None:
        base_conf_thresholds = {
            "wall": 0.25,
            "door": 0.25,
            "window": 0.25,
            "room": 0.25,
        }
    
    logger.info("Starting auto-tune...")
    
    # Resize image for quick pre-pass (use smaller version for speed)
    h, w = image.shape[:2]
    scale_factor = min(640 / max(h, w), 1.0)
    if scale_factor < 1.0:
        import cv2
        small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        test_image = cv2.resize(image, (small_w, small_h))
    else:
        test_image = image
    
    best_score = -1
    best_imgsz = candidate_imgsz[0]
    best_confs = base_conf_thresholds.copy()
    
    # Test each imgsz candidate
    for imgsz in candidate_imgsz:
        try:
            # Quick inference with base thresholds
            detections = model_manager.predict(
                test_image,
                imgsz=imgsz,
                conf_thresholds=base_conf_thresholds,
            )
            
            # Score based on detection quality
            score = score_detections(detections, test_image.shape)
            
            logger.info(f"imgsz={imgsz}, score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_imgsz = imgsz
                
        except Exception as e:
            logger.warning(f"Error testing imgsz={imgsz}: {e}")
            continue
    
    # Adjust confidence thresholds based on detection density
    detections = model_manager.predict(
        test_image,
        imgsz=best_imgsz,
        conf_thresholds=base_conf_thresholds,
    )
    
    # Analyze detection statistics
    stats = analyze_detection_stats(detections)
    
    # Adjust thresholds
    adjusted_confs = adjust_thresholds(base_conf_thresholds, stats)
    
    logger.info(f"Auto-tune complete: imgsz={best_imgsz}, confs={adjusted_confs}")
    
    return {
        "imgsz": best_imgsz,
        "conf_thresholds": adjusted_confs,
    }


def score_detections(detections: Dict[str, List[Dict]], image_shape: Tuple[int, int]) -> float:
    """
    Score detection quality.
    Higher score = better quality.
    """
    h, w = image_shape[:2]
    total_area = h * w
    
    score = 0.0
    
    # Wall continuity score
    walls = detections.get("wall", [])
    if walls:
        wall_coverage = sum(
            (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
            for det in walls
        ) / total_area
        # Prefer moderate coverage (not too sparse, not too dense)
        wall_score = 1.0 - abs(wall_coverage - 0.1)  # Target ~10% coverage
        score += wall_score * 0.4
    
    # Room closure score
    rooms = detections.get("room", [])
    if rooms:
        room_count = len(rooms)
        # Prefer 2-10 rooms for typical floor plans
        if 2 <= room_count <= 10:
            room_score = 1.0
        elif room_count > 10:
            room_score = max(0, 1.0 - (room_count - 10) * 0.1)
        else:
            room_score = room_count * 0.5
        score += room_score * 0.3
    
    # Door/window presence
    doors = detections.get("door", [])
    windows = detections.get("window", [])
    if doors or windows:
        score += 0.2
    
    # Detection confidence average
    all_detections = []
    for class_dets in detections.values():
        all_detections.extend(class_dets)
    
    if all_detections:
        avg_conf = np.mean([det["confidence"] for det in all_detections])
        score += avg_conf * 0.1
    
    return score


def analyze_detection_stats(detections: Dict[str, List[Dict]]) -> Dict:
    """Analyze detection statistics."""
    stats = {}
    
    for class_name, class_dets in detections.items():
        if not class_dets:
            stats[class_name] = {
                "count": 0,
                "avg_confidence": 0.0,
                "density": "sparse",
            }
            continue
        
        count = len(class_dets)
        avg_conf = np.mean([det["confidence"] for det in class_dets])
        
        # Determine density
        if count < 5:
            density = "sparse"
        elif count > 50:
            density = "noisy"
        else:
            density = "normal"
        
        stats[class_name] = {
            "count": count,
            "avg_confidence": float(avg_conf),
            "density": density,
        }
    
    return stats


def adjust_thresholds(
    base_thresholds: Dict[str, float],
    stats: Dict,
) -> Dict[str, float]:
    """Adjust confidence thresholds based on statistics."""
    adjusted = base_thresholds.copy()
    
    for class_name, stat in stats.items():
        if class_name not in adjusted:
            continue
        
        base_conf = adjusted[class_name]
        density = stat.get("density", "normal")
        avg_conf = stat.get("avg_confidence", base_conf)
        
        if density == "noisy":
            # Increase threshold to filter noise
            adjusted[class_name] = min(1.0, base_conf + 0.1)
        elif density == "sparse":
            # Decrease threshold to catch more
            adjusted[class_name] = max(0.1, base_conf - 0.05)
        else:
            # Normal density, slight adjustment based on avg confidence
            if avg_conf > 0.7:
                adjusted[class_name] = min(1.0, base_conf + 0.05)
            elif avg_conf < 0.4:
                adjusted[class_name] = max(0.1, base_conf - 0.05)
    
    return adjusted

