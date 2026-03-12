"""
PDF to Image Conversion Module
Handles multi-page PDF conversion to images
"""

import os
import cv2
import numpy as np
from pathlib import Path

try:
    import pypdfium2 as pdfium
    PDF_BACKEND = "pypdfium2"
except ImportError:
    try:
        from pdf2image import convert_from_path
        PDF_BACKEND = "pdf2image"
    except ImportError:
        PDF_BACKEND = None

class PDFProcessor:
    """Convert PDF documents to images"""
    
    def __init__(self, dpi=300):
        """
        Args:
            dpi: Resolution for image conversion (default: 300)
        """
        self.dpi = dpi
        
        if PDF_BACKEND is None:
            raise ImportError(
                "No PDF backend available. Install pypdfium2 or pdf2image"
            )
        
        print(f"  Using PDF backend: {PDF_BACKEND}")
    
    def pdf_to_images(self, pdf_path):
        """
        Convert PDF to list of images (one per page)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of numpy arrays (images in BGR format for OpenCV)
        """
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if PDF_BACKEND == "pypdfium2":
            return self._convert_with_pypdfium(pdf_path)
        else:
            return self._convert_with_pdf2image(pdf_path)
    
    def _convert_with_pypdfium(self, pdf_path):
        """Convert using pypdfium2 (faster, no poppler dependency)"""
        
        images = []
        pdf = pdfium.PdfDocument(pdf_path)
        
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            
            # Render page to PIL image
            pil_image = page.render(
                scale=self.dpi/72,  # 72 DPI is default
                rotation=0,
            ).to_pil()
            
            # Convert PIL to OpenCV format (BGR)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            images.append(image)
        
        pdf.close()
        return images
    
    def _convert_with_pdf2image(self, pdf_path):
        """Convert using pdf2image (requires poppler)"""
        
        pil_images = convert_from_path(pdf_path, dpi=self.dpi)
        
        # Convert to OpenCV format
        images = []
        for pil_img in pil_images:
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            images.append(image)
        
        return images
    
    def save_images(self, images, output_dir, prefix="page"):
        """
        Save images to directory
        
        Args:
            images: List of image arrays
            output_dir: Directory to save images
            prefix: Filename prefix (default: "page")
            
        Returns:
            List of saved file paths
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for idx, image in enumerate(images):
            filename = f"{prefix}_{idx+1:03d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, image)
            saved_paths.append(filepath)
        
        return saved_paths
