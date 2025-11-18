"""File management for temporary uploads and outputs."""
import logging
import shutil
import time
from pathlib import Path
from typing import Optional
import uuid

logger = logging.getLogger(__name__)


class FileManager:
    """Manages temporary files for requests."""
    
    def __init__(self, temp_dir: str, ttl_seconds: int = 3600):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
    
    def create_request_dir(self, request_id: Optional[str] = None) -> Path:
        """Create a directory for a request."""
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        request_dir = self.temp_dir / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        return request_dir
    
    def get_request_dir(self, request_id: str) -> Optional[Path]:
        """Get directory for a request if it exists."""
        request_dir = self.temp_dir / request_id
        if request_dir.exists():
            return request_dir
        return None
    
    def cleanup_old_files(self):
        """Remove files older than TTL."""
        current_time = time.time()
        removed_count = 0
        
        for item in self.temp_dir.iterdir():
            if not item.is_dir():
                continue
            
            # Check modification time
            try:
                mtime = item.stat().st_mtime
                age = current_time - mtime
                
                if age > self.ttl_seconds:
                    shutil.rmtree(item)
                    removed_count += 1
                    logger.info(f"Cleaned up old request directory: {item.name}")
            except Exception as e:
                logger.warning(f"Error cleaning up {item}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old request directories")
    
    def save_file(self, request_dir: Path, filename: str, content: bytes):
        """Save file to request directory."""
        file_path = request_dir / filename
        file_path.write_bytes(content)
        return file_path
    
    def get_file_path(self, request_dir: Path, filename: str) -> Optional[Path]:
        """Get path to file in request directory."""
        file_path = request_dir / filename
        if file_path.exists():
            return file_path
        return None

