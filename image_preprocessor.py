"""Image preprocessing - MINIMAL version"""

import cv2
import numpy as np


class ImagePreprocessor:
    """Minimal enhancement for VLM"""
    
    def preprocess_for_vlm(self, image):
        """Just sharpen slightly, nothing else"""
        try:
            # Gentle sharpening only
            kernel = np.array([[-0.5, -0.5, -0.5],
                              [-0.5,  5.0, -0.5],
                              [-0.5, -0.5, -0.5]])
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend: 70% original + 30% sharpened
            result = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            
            return result
            
        except Exception as e:
            return image  # Fallback to original

