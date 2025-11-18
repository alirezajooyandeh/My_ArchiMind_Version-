"""Tests for geometry utilities."""
import pytest
from backend.geometry import (
    calculate_area_sf,
    polygon_to_wkt,
)
from backend.scale import (
    calculate_pixels_per_foot_from_points,
    resolve_scale,
)


def test_calculate_area_sf():
    """Test area conversion from pixels to square feet."""
    # 1000 pixels at 10 pixels per foot = 10 square feet
    assert calculate_area_sf(1000, 10) == 10.0
    
    # 100 pixels at 5 pixels per foot = 4 square feet
    assert calculate_area_sf(100, 5) == 4.0
    
    # Invalid pixels_per_foot
    assert calculate_area_sf(100, 0) is None
    assert calculate_area_sf(100, -1) is None


def test_polygon_to_wkt():
    """Test polygon to WKT conversion."""
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    wkt = polygon_to_wkt(polygon)
    
    assert "POLYGON" in wkt
    assert "0 0" in wkt
    assert "10 10" in wkt
    
    # Empty polygon
    assert polygon_to_wkt([]) == ""
    assert polygon_to_wkt([[0, 0]]) == ""


def test_calculate_pixels_per_foot_from_points():
    """Test pixels per foot calculation from two points."""
    # 100 pixel distance, 10 feet = 10 pixels per foot
    point1 = (0, 0)
    point2 = (100, 0)
    assert calculate_pixels_per_foot_from_points(point1, point2, 10) == 10.0
    
    # Diagonal distance
    point1 = (0, 0)
    point2 = (30, 40)  # 50 pixel distance
    result = calculate_pixels_per_foot_from_points(point1, point2, 5)
    assert abs(result - 10.0) < 0.01
    
    # Invalid distance
    assert calculate_pixels_per_foot_from_points(point1, point2, 0) is None
    assert calculate_pixels_per_foot_from_points(point1, point1, 10) is None


def test_resolve_scale():
    """Test scale resolution."""
    # Unknown mode
    assert resolve_scale("unknown") is None
    
    # Explicit mode
    assert resolve_scale("explicit", explicit_scale=10.0) == 10.0
    assert resolve_scale("explicit", explicit_scale=None) is None
    assert resolve_scale("explicit", explicit_scale=0) is None
    
    # Bar mode
    point1 = (0, 0)
    point2 = (100, 0)
    result = resolve_scale("bar", measure_points=(point1, point2), measure_distance_ft=10)
    assert result == 10.0
    
    # Bar mode without points
    assert resolve_scale("bar") is None

