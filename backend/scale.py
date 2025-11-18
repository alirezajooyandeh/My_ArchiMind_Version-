"""Scale estimation and conversion utilities."""
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def calculate_pixels_per_foot_from_points(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    real_distance_ft: float,
) -> Optional[float]:
    """
    Calculate pixels per foot from two clicked points and real-world distance.
    
    Args:
        point1: (x, y) coordinates of first point
        point2: (x, y) coordinates of second point
        real_distance_ft: Real-world distance in feet
    
    Returns:
        Pixels per foot, or None if invalid
    """
    if real_distance_ft <= 0:
        return None
    
    # Calculate pixel distance
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    pixel_distance = (dx ** 2 + dy ** 2) ** 0.5
    
    if pixel_distance <= 0:
        return None
    
    pixels_per_foot = pixel_distance / real_distance_ft
    return pixels_per_foot


def resolve_scale(
    scale_mode: str,
    explicit_scale: Optional[float] = None,
    measure_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    measure_distance_ft: Optional[float] = None,
) -> Optional[float]:
    """
    Resolve pixels per foot based on scale mode.
    
    Args:
        scale_mode: "unknown", "explicit", or "bar"
        explicit_scale: Pixels per foot (for explicit mode)
        measure_points: Tuple of two points (for bar mode)
        measure_distance_ft: Real distance in feet (for bar mode)
    
    Returns:
        Pixels per foot, or None if unknown
    """
    if scale_mode == "explicit":
        if explicit_scale and explicit_scale > 0:
            return explicit_scale
        else:
            logger.warning("explicit scale mode but no valid explicit_scale provided")
            return None
    
    elif scale_mode == "bar":
        if measure_points and measure_distance_ft:
            point1, point2 = measure_points
            return calculate_pixels_per_foot_from_points(point1, point2, measure_distance_ft)
        else:
            logger.warning("bar scale mode but measure points/distance not provided")
            return None
    
    else:  # unknown
        return None

