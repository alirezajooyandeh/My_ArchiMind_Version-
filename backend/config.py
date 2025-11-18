"""Configuration management for the floor plan analysis app."""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model weights paths
    wall_weights_path: Optional[str] = None
    door_weights_path: Optional[str] = None
    window_weights_path: Optional[str] = None
    room_weights_path: Optional[str] = None
    
    # Default detection parameters
    default_imgsz: int = 1280
    default_wall_conf: float = 0.25
    default_door_conf: float = 0.25
    default_window_conf: float = 0.25
    default_room_conf: float = 0.25
    
    # Server configuration
    max_upload_size_mb: int = 50
    temp_dir: str = "./temp"
    temp_ttl_seconds: int = 3600
    host: str = "127.0.0.1"
    port: int = 8000
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda
    
    # Logging
    log_level: str = "INFO"
    debug_mode: bool = False
    
    # Wall rendering configuration
    render_walls_as_boxes: bool = True
    disable_legacy_wall_overlay: bool = True
    disable_wall_labels: bool = True  # never print "Wall 0.xx"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure temp directory exists
Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)

# Wall rendering flags (for easy import)
RENDER_WALLS_AS_BOXES = settings.render_walls_as_boxes
DISABLE_LEGACY_WALL_OVERLAY = settings.disable_legacy_wall_overlay
DISABLE_WALL_LABELS = settings.disable_wall_labels

