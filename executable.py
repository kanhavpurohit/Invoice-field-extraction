"""
IDFC Hackathon - Document Field Extraction
Main executable file for processing invoice PDFs
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.pdf_processor import PDFProcessor
from utils.vlm_extractor import Qwen2VLExtractor
from utils.yolo_detector import SignatureStampDetector
from utils.ensemble import DualPathExtractor


class DocumentExtractor:
    """Main pipeline for document field extraction"""
    
    def __init__(self, model_dir="./models"):
        print("="*60)
        print("IDFC Document AI - Initializing Pipeline")
        print("="*60)
        
        self.model_dir = model_dir
        
        # Initialize components
        print("\n[1/4] Loading PDF processor...")
        self.pdf_processor = PDFProcessor()
        
        print("[2/4] Loading Nanonets VLM (this may take 30-60 seconds)...")
        self.vlm = Qwen2VLExtractor(
            model_path=model_dir,
            use_4bit=True  # 4-bit for speed
        )
        
        print("[3/4] Loading YOLO detector...")
        self.yolo = SignatureStampDetector(
            weights_dir=os.path.join(model_dir, "yolo_weights")
        )
        
        print("[4/4] Initializing dual-path ensemble...")
        self.ensemble = DualPathExtractor(
            qwen_vlm=self.vlm,
            yolo=self.yolo,
            got_ocr_path=os.path.join(model_dir, "got_ocr")
        )
        
        print("\n" + "="*60)
        print("✓ Pipeline Ready!")
        print("="*60 + "\n")
    
    def process_pdf(self, pdf_path, output_dir="./outputs"):
        """
        Process a PDF document and extract fields from all pages
        
        Args:
            pdf_path: Path to input PDF
            output_dir: Directory to save results
        
        Returns:
            List of results (one per page)
        """
        print(f"\nProcessing PDF: {pdf_path}")
        start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert PDF to images
        print("Converting PDF to images...")
        images = self.pdf_processor.pdf_to_images(pdf_path)
        print(f"✓ Extracted {len(images)} pages")
        
        # Process each page
        results = []
        for idx, image in enumerate(images):
            page_num = idx + 1
            print(f"\n--- Processing Page {page_num}/{len(images)} ---")
            page_start = time.time()
            
            # Extract fields using ensemble
            result = self.ensemble.extract_fields(image)
            
            # Add metadata
            result["doc_id"] = f"{Path(pdf_path).stem}_page_{page_num}"
            result["page_number"] = page_num
            result["processing_time_sec"] = round(time.time() - page_start, 2)
            result["cost_estimate_usd"] = self._calculate_cost(result["processing_time_sec"])
            
            results.append(result)
            
            # Print summary
            self._print_result_summary(result)
        
        # Save results
        output_file = os.path.join(
            output_dir,
            f"{Path(pdf_path).stem}_results.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Processing Complete!")
        print(f"  Total pages: {len(results)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time/page: {total_time/len(results):.2f}s")
        print(f"  Results saved: {output_file}")
        print(f"{'='*60}\n")
        
        return results
    
    def process_single_image(self, image_path, output_dir="./outputs"):
        """
        Process a single image file
        
        Args:
            image_path: Path to image file
            output_dir: Directory to save results
        
        Returns:
            Extraction result dictionary
        """
        print(f"\nProcessing Image: {image_path}")
        start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image with error handling
        import cv2
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        print(f"  Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Extract fields
        result = self.ensemble.extract_fields(image)
        
        # Add metadata
        result["doc_id"] = Path(image_path).stem
        result["processing_time_sec"] = round(time.time() - start_time, 2)
        result["cost_estimate_usd"] = self._calculate_cost(result["processing_time_sec"])
        
        # Save result
        output_file = os.path.join(
            output_dir,
            f"{Path(image_path).stem}_result.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Processing complete in {result['processing_time_sec']}s")
        print(f"  Results saved: {output_file}")
        
        self._print_result_summary(result)
        
        return result
    
    def _calculate_cost(self, processing_time_sec):
        """Calculate cost estimate based on processing time"""
        # Using Qwen2-VL-2B on GPU (free/open-source)
        # Estimate: $0.001 per image for GPU electricity/cloud cost
        base_cost = 0.001
        
        # Add small cost for longer processing (proportional to time)
        time_cost = (processing_time_sec / 30.0) * 0.001
        
        return round(base_cost + time_cost, 4)
    
    def _print_result_summary(self, result):
        """Print formatted result summary"""
        fields = result.get("fields", {})
        
        print(f"\n  Extracted Fields:")
        print(f"    Dealer Name : {fields.get('dealer_name') or 'N/A'}")
        print(f"    Model Name  : {fields.get('model_name') or 'N/A'}")
        print(f"    Horse Power : {fields.get('horse_power') or 'N/A'}")
        print(f"    Asset Cost  : {fields.get('asset_cost') or 'N/A'}")
        
        # Signature and stamp status
        sig = fields.get('signature', {})
        stamp = fields.get('stamp', {})
        print(f"    Signature   : {sig.get('present', False)}")
        print(f"    Stamp       : {stamp.get('present', False)}")
        print(f"  Confidence    : {result.get('confidence', 0):.2f}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="IDFC Document AI - Extract fields from invoice PDFs"
    )
    parser.add_argument(
        "input_path",
        help="Path to input PDF or image file"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Directory to save results (default: ./outputs)"
    )
    parser.add_argument(
        "--model-dir",
        default="./models",
        help="Directory containing model weights (default: ./models)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        sys.exit(1)
    
    # Initialize extractor
    try:
        extractor = DocumentExtractor(model_dir=args.model_dir)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Check input type
    input_path = args.input_path
    try:
        if input_path.lower().endswith('.pdf'):
            # Process PDF
            results = extractor.process_pdf(input_path, args.output_dir)
        elif input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Process single image
            result = extractor.process_single_image(input_path, args.output_dir)
        else:
            print(f"Error: Unsupported file type: {input_path}")
            print("Supported: .pdf, .png, .jpg, .jpeg")
            sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
