"""
VisionAssist AI - Feature 3: Smart OCR Reader
Owner: Person 3

Transforms:
Camera Image -> Text Extraction -> Key Information Parsing (Name, Dosage, Expiry) -> Actionable Audio Readout
"""

import re
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

_easyocr_reader = None


def get_ocr_reader():
    """Lazy loads EasyOCR English reader."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            # gpu=False ensures instant compatibility across any hackathon machine
            _easyocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            print(f"[SmartOCR] Notice: EasyOCR init: {e}")
            return None
    return _easyocr_reader


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocesses image to enhance text contrast for better OCR quality."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Mild bilateral filtering to reduce noise while keeping text edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    return filtered


def parse_smart_patterns(text_lines: List[str]) -> Dict[str, Any]:
    """
    Applies regex patterns to identify structured information
    such as medicine dosage, expiry date, brand name, and directions.
    """
    full_text = " ".join(text_lines).strip()
    
    # 1. Dosage Pattern: e.g., 500mg, 500 mg, 10ml, 250 mcg, 2 tablets
    dosage_match = re.search(
        r'\b(\d+(?:\.\d+)?)\s*(mg|ml|g|milligrams|micrograms|mcg|tablets?|capsules?)\b',
        full_text,
        re.IGNORECASE
    )
    dosage = dosage_match.group(0) if dosage_match else None

    # 2. Expiry Date Pattern: e.g., EXP 12/2027, Expiry: Dec 2026, EXP: 05-2028
    expiry_match = re.search(
        r'(?:exp|expiry|exp\.?\s*date|use\s*before|best\s*before)[\s:]*([0-9]{1,2}[/\-\.][0-9]{2,4}|[A-Za-z]{3,9}\s+[0-9]{4}|[0-9]{2,4}[/\-\.][0-9]{1,2})',
        full_text,
        re.IGNORECASE
    )
    expiry = expiry_match.group(1).strip() if expiry_match else None

    # Secondary loose check for date formatted after "EXP" or standing alone as date
    if not expiry:
        date_standalone = re.search(r'\b(0[1-9]|1[0-2])[-/](20\d{2}|\d{2})\b', full_text)
        if date_standalone:
            expiry = date_standalone.group(0)

    # 3. Product / Medicine Name heuristic: Take prominent first line or words before dosage
    product_name = None
    if text_lines:
        first_clean_line = text_lines[0].strip()
        # If line contains dosage, strip or extract name
        if dosage and dosage in first_clean_line:
            product_name = first_clean_line.replace(dosage, "").strip(" -:,")
        elif len(first_clean_line) > 2:
            product_name = first_clean_line

    # Format actionable speech readout
    speech_parts = []
    
    if product_name and len(product_name) > 1:
        if dosage:
            speech_parts.append(f"This is {product_name} {dosage}.")
        else:
            speech_parts.append(f"Product label reads: {product_name}.")
    elif dosage:
        speech_parts.append(f"Dosage is {dosage}.")

    if expiry:
        speech_parts.append(f"Expiry date: {expiry}.")

    # Fallback if no specific structured fields detected
    if not speech_parts:
        if full_text:
            # Speak first 250 characters of readable text
            short_text = full_text[:250]
            speech_parts.append(f"The text reads: {short_text}")
        else:
            speech_parts.append("No clear text detected in this image.")

    spoken_response = " ".join(speech_parts)

    return {
        "full_text": full_text,
        "product_name": product_name,
        "dosage": dosage,
        "expiry": expiry,
        "spoken_response": spoken_response
    }


def read_text(image: np.ndarray) -> Dict[str, Any]:
    """
    Core function for Feature 3.
    Extracts text from the image, matches smart patterns, and returns
    both structured metadata and an actionable spoken response.
    """
    if image is None:
        return {
            "success": False,
            "spoken_response": "No image was provided to read text from.",
            "full_text": "",
            "extracted_fields": {}
        }

    reader = get_ocr_reader()
    text_lines = []

    if reader is not None:
        try:
            processed = preprocess_image(image)
            raw_results = reader.readtext(processed, detail=0)
            text_lines = [t.strip() for t in raw_results if t.strip()]
        except Exception as e:
            print(f"[SmartOCR] OCR read error: {e}")

    # Fallback if EasyOCR didn't find anything or wasn't available
    if not text_lines:
        # Check if tesseract is installed
        try:
            import pytesseract
            raw_str = pytesseract.image_to_string(image)
            text_lines = [line.strip() for line in raw_str.split("\n") if line.strip()]
        except Exception:
            pass

    if not text_lines:
        return {
            "success": False,
            "spoken_response": "I could not detect any readable text on this item.",
            "full_text": "",
            "extracted_fields": {}
        }

    parsed = parse_smart_patterns(text_lines)

    return {
        "success": True,
        "spoken_response": parsed["spoken_response"],
        "full_text": parsed["full_text"],
        "lines": text_lines,
        "extracted_fields": {
            "product_name": parsed["product_name"],
            "dosage": parsed["dosage"],
            "expiry": parsed["expiry"]
        }
    }


# Standalone runner for Person 3 unit testing
if __name__ == "__main__":
    print("[Person 3 - Smart OCR Reader Standalone Test]")
    sample_img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.putText(sample_img, "Paracetamol 500 mg", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(sample_img, "EXP: 12/2027", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    res = read_text(sample_img)
    print("Spoken Response:", res["spoken_response"])
    print("Full Output:", res)
