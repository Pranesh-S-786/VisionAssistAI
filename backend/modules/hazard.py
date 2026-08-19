"""
VisionAssist AI - Feature 2: Context-Aware Hazard Detection
Owner: Person 2

Relevance Filter:
Does NOT announce every peripheral object.
Only triggers audio warnings when an obstacle is close and blocking the central walking path.

Transforms:
Obstacle in path -> Proximity & Direction Calculation -> Actionable Evading Direction
"""

import cv2
import numpy as np
from typing import Dict, Any, List
from .object_finder import get_yolo_model

# Obstacle classes that can impede indoor navigation
HAZARD_CLASSES = {
    "person", "chair", "dining table", "couch", "sofa",
    "bed", "tv", "refrigerator", "potted plant", "suitcase",
    "backpack", "box", "bench", "door"
}


def check_hazards(image: np.ndarray, model=None) -> Dict[str, Any]:
    """
    Core function for Feature 2.
    Evaluates obstacles in the user's immediate path.
    Returns clear, actionable voice alert if a path hazard is detected,
    or confirmation that the path is clear.
    """
    if image is None:
        return {
            "hazard": False,
            "spoken_response": "No camera feed available for hazard check.",
            "hazards_detected": []
        }

    active_model = model or get_yolo_model()
    if active_model is None:
        return {
            "hazard": False,
            "spoken_response": "Hazard detection model is currently unavailable.",
            "hazards_detected": []
        }

    h, w = image.shape[:2]
    img_area = max(h * w, 1)

    results = active_model(image, conf=0.25, verbose=False)[0]

    hazards_in_path: List[Dict[str, Any]] = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = active_model.names[cls_id].lower()
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Check if class is a navigational obstacle
        if cls_name not in HAZARD_CLASSES:
            continue

        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0

        area_ratio = box_area / img_area
        height_ratio = box_height / h

        # Centrality check: Is the obstacle in the central 60% walking zone (20% to 80% width)?
        is_central = (0.20 * w) < x_center < (0.80 * w)

        # Proximity proxy: Large box area (> 10% of frame) or substantial height (> 30% of frame)
        is_close = (area_ratio > 0.10) or (height_ratio > 0.30)

        # Bottom proximity: Is the base of the object near the lower half of screen (ground level)?
        is_ground_near = (y2 > 0.50 * h)

        if is_central and is_close and is_ground_near:
            # Determine evasion guidance direction
            if x_center > (0.50 * w):
                # Obstacle is towards right of center -> prompt user to step left
                evade_dir = "left"
                guidance = f"{cls_name.capitalize()} ahead. Move slightly left."
            elif x_center < (0.50 * w):
                # Obstacle is towards left of center -> prompt user to step right
                evade_dir = "right"
                guidance = f"{cls_name.capitalize()} ahead. Move slightly right."
            else:
                evade_dir = "around"
                guidance = f"Warning: {cls_name.capitalize()} directly in front of you. Please stop or step aside."

            hazards_in_path.append({
                "class": cls_name,
                "confidence": round(conf, 2),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "area_ratio": round(area_ratio, 3),
                "evade_direction": evade_dir,
                "guidance": guidance
            })

    if hazards_in_path:
        # Choose the most critical hazard (largest area / closest)
        primary_hazard = max(hazards_in_path, key=lambda hz: hz["area_ratio"])
        return {
            "hazard": True,
            "spoken_response": primary_hazard["guidance"],
            "primary_hazard": primary_hazard,
            "hazards_detected": hazards_in_path
        }
    else:
        return {
            "hazard": False,
            "spoken_response": "Path ahead appears clear.",
            "hazards_detected": []
        }


# Standalone runner for Person 2 unit testing
if __name__ == "__main__":
    print("[Person 2 - Hazard Detection Standalone Test]")
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    res = check_hazards(test_img)
    print("Result:", res)
