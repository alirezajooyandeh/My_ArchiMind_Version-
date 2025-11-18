"""YOLO model loading and inference."""
import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages YOLO models for different classes."""
    
    def __init__(self, settings):
        self.settings = settings
        self.models: Dict[str, Optional[YOLO]] = {}
        self.device = self._detect_device()
        self._load_models()
    
    def _detect_device(self) -> str:
        """Detect and return the appropriate device."""
        device_override = self.settings.device.lower()
        
        if device_override == "cpu":
            return "cpu"
        elif device_override == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        else:  # auto
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
    
    def _load_models(self):
        """Load YOLO models for each class."""
        class_configs = {
            "wall": self.settings.wall_weights_path,
            "door": self.settings.door_weights_path,
            "window": self.settings.window_weights_path,
            "room": self.settings.room_weights_path,
        }
        
        for class_name, weights_path in class_configs.items():
            if weights_path and Path(weights_path).exists():
                try:
                    logger.info(f"Loading {class_name} model from {weights_path}")
                    model = YOLO(weights_path)
                    model.to(self.device)
                    self.models[class_name] = model
                    logger.info(f"Successfully loaded {class_name} model on {self.device}")
                except Exception as e:
                    logger.error(f"Failed to load {class_name} model: {e}")
                    self.models[class_name] = None
            else:
                logger.warning(f"No weights path provided or file not found for {class_name}")
                self.models[class_name] = None
    
    def get_available_classes(self) -> List[str]:
        """Return list of classes with loaded models."""
        return [cls for cls, model in self.models.items() if model is not None]
    
    def predict(
        self,
        image: np.ndarray,
        imgsz: int = 1280,
        conf_thresholds: Optional[Dict[str, float]] = None,
        nms_iou_thresholds: Optional[Dict[str, float]] = None,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Run inference on image.
        
        Returns:
            Dictionary mapping class names to lists of detection dicts.
            Each detection dict contains: bbox, confidence, class_name
        """
        if conf_thresholds is None:
            conf_thresholds = {}
        
        if nms_iou_thresholds is None:
            nms_iou_thresholds = {}
        
        if classes is None:
            classes = self.get_available_classes()
        
        results = {}
        
        for class_name in classes:
            if class_name not in self.models or self.models[class_name] is None:
                continue
            
            model = self.models[class_name]
            conf = conf_thresholds.get(class_name, 0.25)
            iou = nms_iou_thresholds.get(class_name, 0.45)  # Default NMS IoU threshold
            
            try:
                # Run inference with NMS IoU threshold
                yolo_results = model.predict(
                    image,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,  # NMS IoU threshold
                    device=self.device,
                    verbose=False,
                )
                
                detections = []
                if yolo_results and len(yolo_results) > 0:
                    result = yolo_results[0]
                    
                    if result.boxes is not None:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            # Get bounding box coordinates
                            box = boxes.xyxy[i].cpu().numpy()
                            confidence = float(boxes.conf[i].cpu().numpy())
                            
                            # Get segmentation if available (for rooms)
                            polygon = None
                            if result.masks is not None and i < len(result.masks.data):
                                mask = result.masks.data[i].cpu().numpy()
                                # Convert mask to polygon
                                polygon = self._mask_to_polygon(mask)
                            
                            detections.append({
                                "bbox": box.tolist(),  # [x_min, y_min, x_max, y_max]
                                "confidence": confidence,
                                "class_name": class_name,
                                "polygon": polygon,
                            })
                
                results[class_name] = detections
                
            except Exception as e:
                logger.error(f"Error during inference for {class_name}: {e}")
                results[class_name] = []
        
        return results
    
    def _mask_to_polygon(self, mask: np.ndarray) -> Optional[List[List[float]]]:
        """Convert binary mask to polygon coordinates."""
        try:
            from shapely.geometry import Polygon
            import cv2
            
            # Find contours
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Simplify polygon
            epsilon = 0.002 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Convert to list of [x, y] pairs
            polygon = [[float(point[0][0]), float(point[0][1])] for point in approx]
            
            return polygon if len(polygon) >= 3 else None
            
        except Exception as e:
            logger.error(f"Error converting mask to polygon: {e}")
            return None

