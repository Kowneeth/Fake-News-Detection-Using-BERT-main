"""
Multi-Modal Image & Meme Verification Forensics.
Extracts EXIF metadata signatures, flags image editing software tampering (Photoshop, Canva),
and extracts text from memes and screenshot captures.
"""
import io
import re
import base64
import logging
from typing import Dict, Any, Optional
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

# Known digital image manipulation software tags
EDITING_SOFTWARE_SIGNATURES = [
    "photoshop", "gimp", "canva", "snapseed", "picsart",
    "lightroom", "affinity", "pixelmator", "paint.net"
]


class MultiModalVerifier:
    """
    Performs forensic metadata inspection and text extraction on uploaded images.
    """
    def analyze_image(self, image_bytes: bytes, filename: str = "upload.jpg") -> Dict[str, Any]:
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image format: {e}")

        width, height = image.size
        img_format = image.format or "JPEG"

        # 1. EXIF Metadata Extraction
        exif_data = {}
        software_detected = ""
        is_manipulated = False
        date_original = "Not recorded"
        date_modified = "Not recorded"

        try:
            raw_exif = image._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, (str, int, float)):
                        exif_data[tag_name] = str(value)

                # Check software signature
                software_val = exif_data.get("Software", "").lower()
                for sig in EDITING_SOFTWARE_SIGNATURES:
                    if sig in software_val:
                        software_detected = exif_data.get("Software", "")
                        is_manipulated = True
                        break

                date_original = exif_data.get("DateTimeOriginal", "Not recorded")
                date_modified = exif_data.get("DateTime", "Not recorded")
        except Exception as e:
            logger.debug(f"Could not read EXIF tags: {e}")

        # Metadata stripped evaluation (social media re-uploads strip EXIF)
        metadata_stripped = (len(exif_data) == 0)

        # 2. Tampering Risk Assessment
        if is_manipulated:
            tampering_risk = "ELEVATED (EDITING SOFTWARE SIGNATURE PRESENT)"
        elif metadata_stripped:
            tampering_risk = "MODERATE (METADATA STRIPPED / SOCIAL COMPRESSION)"
        else:
            tampering_risk = "LOW TAMPERING RISK"

        # 3. Text Extraction (OCR / Meme text extraction)
        extracted_text = ""
        try:
            import pytesseract
            extracted_text = pytesseract.image_to_string(image).strip()
        except (ImportError, Exception):
            # Fallback simulated extraction for environments without tesseract-ocr binary
            extracted_text = ""

        return {
            "filename": filename,
            "format": img_format,
            "dimensions": f"{width}x{height}",
            "editing_software_detected": is_manipulated,
            "software_name": software_detected or "None detected",
            "metadata_stripped": metadata_stripped,
            "date_original": date_original,
            "date_modified": date_modified,
            "tampering_risk": tampering_risk,
            "extracted_text": extracted_text,
            "has_embedded_text": bool(len(extracted_text) > 5)
        }
