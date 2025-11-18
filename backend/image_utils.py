"""Image and PDF processing utilities."""
import logging
import io
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageOps
import cv2
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError

logger = logging.getLogger(__name__)


def process_uploaded_file(
    file_content: bytes,
    filename: str,
    pdf_dpi: int = 300,
    max_dimension: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Process uploaded file (PDF or image) and return as numpy array.
    
    Returns:
        Tuple of (image_array, metadata_dict)
    """
    file_ext = Path(filename).suffix.lower()
    metadata = {"filename": filename, "original_format": file_ext}
    
    try:
        if file_ext == ".pdf":
            # Process PDF
            image = process_pdf(file_content, pdf_dpi)
            metadata["source"] = "pdf"
            metadata["pdf_dpi"] = pdf_dpi
        elif file_ext in [".jpg", ".jpeg", ".png"]:
            # Process image
            image = process_image(file_content)
            metadata["source"] = "image"
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Apply max dimension constraint
        if max_dimension:
            image = resize_with_aspect_ratio(image, max_dimension)
        
        h, w = image.shape[:2]
        metadata["width"] = w
        metadata["height"] = h
        
        return image, metadata
        
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        raise


def process_pdf(file_content: bytes, dpi: int = 300) -> np.ndarray:
    """Convert first page of PDF to numpy array."""
    try:
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # Convert first page
            images = convert_from_path(tmp_path, dpi=dpi, first_page=1, last_page=1)
            
            if not images:
                raise ValueError("PDF conversion produced no images")
            
            # Convert PIL to numpy
            pil_image = images[0]
            image_array = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV compatibility
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_array
            
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
    except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
        logger.error(f"PDF processing error: {e}")
        raise ValueError(f"Invalid or corrupted PDF: {e}")
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise


def process_image(file_content: bytes) -> np.ndarray:
    """Process uploaded image file."""
    try:
        # Open with PIL to handle EXIF orientation
        pil_image = Image.open(io.BytesIO(file_content))
        
        # Auto-rotate based on EXIF
        pil_image = ImageOps.exif_transpose(pil_image)
        
        # Convert to RGB if needed
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise ValueError(f"Invalid image file: {e}")


def resize_with_aspect_ratio(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """Resize image maintaining aspect ratio, capping longest side at max_dimension."""
    h, w = image.shape[:2]
    
    if max(h, w) <= max_dimension:
        return image
    
    if h > w:
        new_h = max_dimension
        new_w = int(w * (max_dimension / h))
    else:
        new_w = max_dimension
        new_h = int(h * (max_dimension / w))
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def prepare_for_inference(image: np.ndarray, imgsz: int) -> Tuple[np.ndarray, dict]:
    """
    Prepare image for YOLO inference.
    
    Returns:
        Tuple of (prepared_image, transform_metadata)
    """
    h, w = image.shape[:2]
    
    # YOLO handles resizing internally, but we track original dimensions
    transform_meta = {
        "original_width": w,
        "original_height": h,
        "imgsz": imgsz,
    }
    
    return image, transform_meta

