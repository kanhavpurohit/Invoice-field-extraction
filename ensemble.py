import cv2
import re
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer


class DualPathExtractor:
    def __init__(self, qwen_vlm, yolo, got_ocr_path="./models/got_ocr"):
        self.qwen_vlm = qwen_vlm
        self.yolo = yolo
        self.got_model = None
        self.got_tokenizer = None
        
        # SPEED: Disable GOT-OCR by default (saves 10-15 seconds)
        # Uncomment below to enable GOT-OCR validation
        
        # GOT-OCR DISABLED - compatibility issue
        self.got_model = None
        self.got_tokenizer = None
        print("  ✓ VLM-only mode (fast & stable)")

        
        
    
    def extract_fields(self, image):
        """Extract with proper confidence calculation and JSON formatting"""
        print("  Qwen2-VL extraction...")
        
        # SPEED: Resize large images (saves 5-10 seconds for large images)
        h, w = image.shape[:2]
        max_dim = 1280  # Reduce from original size
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"    Resized image from {w}x{h} to {new_w}x{new_h}")
        
        # Get VLM results
        vlm_results = self.qwen_vlm.extract_fields(image)
        
        print("  YOLO detection...")
        # Get YOLO detections with bboxes
        sig_box = self.yolo.detect_signature(image)
        stamp_box = self.yolo.detect_stamp(image)
        
        # SPEED: Skip GOT-OCR for faster processing
        # Uncomment below to enable validation
        got_results = {}   
        
                # IMPROVED CONFIDENCE CALCULATION
        # Use VLM's field-level confidence as base
        vlm_confidence = vlm_results.get('confidence', 0.0)
        
        # Calculate field completeness
        field_keys = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        filled_fields = sum(1 for k in field_keys 
                           if vlm_results.get(k) not in [None, "", 0])
        completeness_score = filled_fields / len(field_keys)
        
        # IMPROVED CONFIDENCE CALCULATION
            # Use VLM's field-level confidence as base
        vlm_confidence = vlm_results.get('confidence', 0.0)

            # Calculate field completeness
        field_keys = ['dealer_name', 'model_name', 'horse_power', 'asset_cost']
        filled_fields = sum(1 for k in field_keys 
                        if vlm_results.get(k) not in [None, "", 0])
        completeness_score = filled_fields / len(field_keys)

            # Penalize for suspicious values
        penalties = 0.0

        
        # Combine: VLM confidence (60%) + completeness (30%) - penalties
        final_confidence = (
            vlm_confidence * 0.6 + 
            completeness_score * 0.3 - 
            penalties
        )
        
        # Clamp to [0, 1]
        final_confidence = max(0.0, min(1.0, final_confidence))
        # SMART ENSEMBLE: Merge VLM + GOT-OCR results
        if got_results:
            
    
            # Model name: Use GOT-OCR if VLM failed or found dealer subtitle
            if got_results.get('model_name'):
                vlm_model = str(vlm_results.get('model_name', ''))
                got_model = str(got_results.get('model_name', ''))
    
                # If VLM model is None/empty, use GOT-OCR
                if not vlm_model or vlm_model.lower() in ['none', 'null']:
                    print(f"  🔄 Using GOT-OCR model name: {got_model}")
                    vlm_results['model_name'] = got_model
                # If VLM found dealer subtitle, prefer GOT-OCR
                elif any(word in vlm_model.lower() for word in ['genuine', 'spares', 'repairs', 'authorized']):
                    print(f"  🔄 Model corrected (VLM found dealer text): '{vlm_model}' → '{got_model}' (GOT-OCR)")
                    vlm_results['model_name'] = got_model
                # If GOT found longer/more complete text
                elif len(got_model) > len(vlm_model) + 5:
                    print(f"  ℹ️ GOT-OCR found longer model name: {got_model}")


        
        # Format according to required JSON structure (CRITICAL!)
        return {
            "fields": {
                "dealer_name": vlm_results.get("dealer_name"),
                "model_name": vlm_results.get("model_name"),
                "horse_power": vlm_results.get("horse_power"),
                "asset_cost": vlm_results.get("asset_cost"),
                "signature": {
                    "present": len(sig_box) > 0,
                    "bbox": sig_box if len(sig_box) > 0 else []
                },
                "stamp": {
                    "present": len(stamp_box) > 0,
                    "bbox": stamp_box if len(stamp_box) > 0 else []
                }
            },
            "confidence": round(final_confidence, 2)
        }
    
    def _extract_with_got(self, image):
        """Extract fields using GOT-OCR - fixed for file path requirement"""
        import tempfile
        import os
    
        tmp_path = None
        cleanup_needed = False
    
        try:
            # GOT-OCR needs a file path, not PIL/numpy
            if isinstance(image, str):
                # Already a file path - use directly
                image_path = image
            else:
                # Need to save to temp file
                cleanup_needed = True
            
                # Convert to PIL if numpy
                if isinstance(image, np.ndarray):
                    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    image_pil = image
            
                # Create temp file
                tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
                os.close(tmp_fd)  # Close file descriptor immediately
            
                # Save image to temp file
                image_pil.save(tmp_path)
                image_path = tmp_path
        
            # Run OCR with file path
            ocr_text = self.got_model.chat(self.got_tokenizer, image_path, ocr_type='ocr')
        
            # Parse fields
            results = {}
        
            # Extract horse power (multiple patterns)
            hp_patterns = [
                r'(?:HP|H\.P\.?|Horse\s*Power)[:\s]*(\d{1,3})',
                r'(\d{2,3})\s*(?:HP|H\.P\.?)',
                r'(\d{2,3})\s*(?:horse\s*power)',
            ]
            for pattern in hp_patterns:
                hp_match = re.search(pattern, ocr_text, re.IGNORECASE)
                if hp_match:
                    hp_val = int(hp_match.group(1))
                    if 15 <= hp_val <= 200:
                        results['horse_power'] = hp_val
                        break
        
            # Extract cost (multiple patterns)
            cost_patterns = [
                r'(?:Total|Grand\s*Total|Net\s*Amount)[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                r'(?:Rs\.?|₹)\s*([0-9,]{6,})',
                r'Total[:\s]*([0-9,]{6,})',
            ]
            for pattern in cost_patterns:
                cost_match = re.search(pattern, ocr_text, re.IGNORECASE)
                if cost_match:
                    cost_str = cost_match.group(1).replace(',', '')
                    try:
                        cost_val = int(cost_str)
                        if 100000 <= cost_val <= 50000000:
                            results['asset_cost'] = cost_val
                            break
                    except:
                        continue
        
            # Extract model name (look for tractor model patterns)
            model_patterns = [
                r'(New\s+Swaraj\s+Tractor\s+\d+\s*\w*)',
                r'(Swaraj\s+\d+\s*\w+)',
                r'(Farmtrac\s+\d+\s+\w+)',
                r'(Sonalika\s+DI\s+\d+\s+\w+)',
                r'(Mahindra\s+\w+\s+\d+)',
                r'(MF\s*[-\s]*\d+\s*\w*)',
                r'(Massey\s+Ferguson\s+\d+\s*\w*)',
                r'(New\s+Holland\s+\d+\s*\w*)',
            ]
            for pattern in model_patterns:
                model_match = re.search(pattern, ocr_text, re.IGNORECASE)
                if model_match:
                    model_name = model_match.group(1).strip()
                    if len(model_name) > 3:
                        results['model_name'] = model_name
                        break
        
            return results
        
        except Exception as e:
            print(f"  GOT-OCR error details: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
        finally:
            # Always clean up temp file if we created one
            if cleanup_needed and tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass



EnsembleExtractor = DualPathExtractor







    
    
    
    
    
    
