"""
YOLO Signature & Stamp Detection Module
Detects signatures and stamps in documents and returns bounding boxes
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


class SignatureStampDetector:
    """Detect signatures and stamps using YOLO"""
    
    def __init__(self, weights_dir="./models/yolo_weights", use_pretrained=True):
        """
        Args:
            weights_dir: Directory for custom YOLO weights
            use_pretrained: Use base YOLOv8 (True) or custom weights (False)
        """
        self.weights_dir = weights_dir
        os.makedirs(weights_dir, exist_ok=True)
        
        if use_pretrained:
            self.model = YOLO('yolov8n.pt')
            print("  Using pretrained YOLOv8-nano")
        else:
            custom_weights = os.path.join(weights_dir, "best.pt")
            if os.path.exists(custom_weights):
                self.model = YOLO(custom_weights)
                print(f"  Using custom weights: {custom_weights}")
            else:
                print(f"  Custom weights not found: {custom_weights}")
                print("  Falling back to pretrained YOLOv8-nano")
                self.model = YOLO('yolov8n.pt')
        
        self.conf_threshold = 0.15
    
    def detect_signature(self, image):
        """Detect signature with improved heuristics"""
        h, w = image.shape[:2]
        
        # SPEED: Resize for faster processing
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]
        
        # Focus on bottom 40% of image
        roi = image[int(h*0.6):, :]
        roi_y_offset = int(h*0.6)
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # SPEED: Skip bilateral filter (saves time, minimal accuracy loss)
        # gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        signature_boxes = []
        for cnt in contours:
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            area = w_cnt * h_cnt
            
            if 300 < area < 80000:
                aspect_ratio = w_cnt / float(h_cnt) if h_cnt > 0 else 0
                if 1.2 < aspect_ratio < 10:
                    roi_cnt = binary[y:y+h_cnt, x:x+w_cnt]
                    density = np.sum(roi_cnt > 0) / (w_cnt * h_cnt)
                    
                    if 0.05 < density < 0.7:
                        signature_boxes.append([
                            x, y + roi_y_offset,
                            x + w_cnt, y + roi_y_offset + h_cnt
                        ])
        
        if signature_boxes:
            largest = max(signature_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            return largest
        
        return []
    
    def detect_stamp(self, image):
        """
        Detect dealer stamp in image (focused on BOTTOM area)
        Args:
            image: OpenCV image (BGR numpy array)
        Returns:
            List of bounding boxes [x1, y1, x2, y2] or empty list
        """
        h, w = image.shape[:2]
    
        # SPEED: Resize for faster processing
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]
    
        # IMPORTANT FIX: Search in BOTTOM 60% where dealer stamps usually are
        # This prevents detecting the logo at top as a stamp
        roi = image[int(h*0.4):, :]  # Bottom 60% of image
        roi_y_offset = int(h*0.4)
        roi_x_offset = 0
    
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
        # Expanded color ranges for stamps (red, blue, green, purple)
        # Red range (common for stamps)
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([180, 255, 255])
    
        # Blue range (very common - like "For Gill Agro")
        lower_blue = np.array([90, 50, 50])  # More lenient blue detection
        upper_blue = np.array([130, 255, 255])
    
        # Green range (less common but exists)
        lower_green = np.array([40, 80, 80])
        upper_green = np.array([80, 255, 255])
    
        # Purple range
        lower_purple = np.array([130, 50, 50])
        upper_purple = np.array([160, 255, 255])
    
        # Create masks for each color
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    
        # Combine all color masks
        mask = mask_red1 | mask_red2 | mask_blue | mask_green | mask_purple
    
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        stamp_boxes = []
        for cnt in contours:
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            area = w_cnt * h_cnt
        
            # Improved stamp heuristics
            if 500 < area < 150000:  # Reasonable size range
                aspect_ratio = w_cnt / float(h_cnt) if h_cnt > 0 else 0
                if 0.3 < aspect_ratio < 4:  # More lenient aspect ratio
                    # Check color density
                    roi_cnt = mask[y:y+h_cnt, x:x+w_cnt]
                    color_density = np.sum(roi_cnt > 0) / (w_cnt * h_cnt)
                
                    if color_density > 0.1:  # At least 10% colored pixels
                        # Add offset to get correct position in original image
                        stamp_boxes.append([
                            x + roi_x_offset,
                            y + roi_y_offset,  # IMPORTANT: Add y offset
                            x + roi_x_offset + w_cnt,
                            y + roi_y_offset + h_cnt
                        ])
    
        # Return largest box
        if stamp_boxes:
            largest = max(stamp_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            return largest
    
        return []

        
    
    def detect_both(self, image):
        """Detect both signature and stamp"""
        sig_box = self.detect_signature(image)
        stamp_box = self.detect_stamp(image)
        
        return {
            "signature": len(sig_box) > 0,
            "stamp": len(stamp_box) > 0,
            "signature_box": sig_box,
            "stamp_box": stamp_box
        }
