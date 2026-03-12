"""
Qwen2-VL Document Extraction Module
Uses Qwen2-VL-2B-Instruct for vision-based field extraction
"""

import json
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class Qwen2VLExtractor:
    """Extract fields using Qwen2-VL-2B vision model"""
    
    def __init__(self, model_path="./models/qwen2_vl_2b", use_4bit=False):
        """
        Args:
            model_path: Path to Qwen2-VL model
            use_4bit: Use 4-bit quantization (for 4GB VRAM)
        """ 
        self.model_path = model_path
        self.use_4bit = use_4bit
        print(f"  Loading Qwen2-VL-2B ({'4-bit quantized' if use_4bit else 'full precision'})...")
        self._load_model()
    
    def _load_model(self):
        """Load Qwen2-VL model"""
        # Check CUDA
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"  ✓ CUDA detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("  ⚠️ WARNING: Using CPU (slower)")
        
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",  # ← Use HuggingFace repo ID
            quantization_config=quantization_config,
            device_map="cuda:0",
            trust_remote_code=True,
            cache_dir=self.model_path  # ← Add this line
        )

        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",  # ← Use HuggingFace repo ID
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
            cache_dir=self.model_path  # ← Add this line
        )

        
        self.processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",  # ← Use HuggingFace repo ID
            trust_remote_code=True,
            cache_dir=self.model_path  # ← Add this line
        )
        # SPEED: Set model to eval mode and enable optimizations
        self.model.eval()
        
        # SPEED: Enable torch.compile if available (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                print("  Enabling torch.compile for faster inference...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"  torch.compile not available: {e}")
        
        print(f"  ✓ Model loaded on {self.device}")
    
    def extract_fields(self, image):
        """Extract fields from image using Qwen2-VL"""
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif isinstance(image, str):
                image = Image.open(image)
            
            # SPEED: Resize if too large (VLM doesn't need ultra-high res)
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                print(f"    Resized to {new_size} for faster inference")
            
            # Create prompt
            prompt = self._create_prompt()
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to GPU
            inputs = inputs.to(self.device)
            
            # SPEED: Generate with optimized settings
            with torch.no_grad():  # Disable gradient computation
                with torch.cuda.amp.autocast():  # Enable mixed precision
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=140,
                        do_sample=False,
                        temperature=0.1,
                        top_p=0.9,
                        num_beams=1,  # No beam search (faster)
                        use_cache=True,  # Enable KV cache
                    )
            
            # Trim and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # DEBUG: Print raw response (comment out for production)
            # print(f"  VLM Response: {response_text[:200]}")
            # DEBUG - Print raw VLM output
            print("="*60)
            print("VLM RAW OUTPUT:")
            print(response_text)
            print("="*60)

            # Parse response
            result = self._parse_response(response_text)
            return result
            
        except Exception as e:
            print(f"  VLM extraction error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result()
        # Parse response
        result = self._parse_response(response_text)

        # OCR FALLBACK: If HP is null, try OCR
        if result.get('horse_power') is None:
            print("  ⚠️ HP not found by VLM, trying OCR fallback...")
            hp_ocr = self._ocr_fallback_hp(image)
            if hp_ocr:
                result['horse_power'] = hp_ocr

        return result

    
    def _create_prompt(self):
        """Optimized prompt for tractor invoices"""
        return """You are extracting data from a tractor invoice/quotation. Return JSON with these fields:

{
  "dealer_name": "shop name",
  "model_name": "tractor model",
  "horse_power": number_or_null,
  "asset_cost": total_amount
}

EXTRACTION GUIDE:

1. DEALER NAME (shop/business selling the tractor):
   - Usually at the very TOP of the document
   - This is the seller's business name, often in regional language
   - Look near: address, phone number, GST number
   - DO NOT extract: tractor manufacturer brands, business taglines, service descriptions

2. MODEL NAME (the tractor model being purchased):
   - Located in the ITEM/PRODUCT section (middle of document)
   - This is listed with a price next to it
   - IMPORTANT: Read ALL text in the model row - both PRINTED and HANDWRITTEN
   - If you see printed brand name followed by handwritten model details, combine them together
   - Give the COMPLETE model name including all numbers, letters, variants
   - DO NOT extract: dealer service descriptions, business slogans, warranty text

For horse_power:
- Look for "HP", "BHP", "Horse Power" anywhere
- Look in specifications, item description, model details
- Extract the NUMBER only (e.g., "48 HP" → 48)
- If not visible → null


4. ASSET COST:
   - The FINAL TOTAL at the BOTTOM of the document
   - Usually labeled: "Total", "Grand Total", "Net Amount", "Amount Payable"
   - This is the largest number at the bottom
   - Extract numeric value only (no currency symbols)

CRITICAL RULES:
- For model_name: Combine printed base name with any handwritten additions
- Read text near item prices to find the actual tractor model
- Dealer info is at TOP, tractor model is in MIDDLE item list
- If multiple tractor options are shown, extract the one with filled-in handwritten details

Return ONLY valid JSON, no additional text."""



    def _parse_response(self, response_text):
        """Parse with dynamic per-field confidence"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
        
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
            else:
                print(f"  No JSON found in response")
                return self._empty_result()
        
            # Calculate DYNAMIC field-level confidence
            field_confidence = {}
        
            # Dealer name confidence
            dealer = str(data.get('dealer_name', ''))
            if not dealer or dealer.lower() in ['none', 'n/a', '']:
                field_confidence['dealer_name'] = 0.0
            elif len(dealer) < 3:  # Too short
                field_confidence['dealer_name'] = 0.5
            elif any(brand in dealer.lower() for brand in ['mahindra', 'sonalika', 'new holland', 'john deere', 'massey']):
                field_confidence['dealer_name'] = 0.6  # Might be brand, not dealer
            else:
                field_confidence['dealer_name'] = 0.95  # Good dealer name
        
            # Model name confidence
            model = str(data.get('model_name', ''))

            # Warn about dealer business descriptions (but don't auto-reject)
            dealer_keywords = ['genuine', 'spares', 'repairs', 'authorized', 'dealer']

            if not model or model.lower() in ['none', 'n/a', '']:
                field_confidence['model_name'] = 0.0
            elif any(keyword in model.lower() for keyword in dealer_keywords):
                # Might be dealer description - lower confidence but keep it
                print(f"  ⚠️ Possible dealer subtitle in model name: '{model}'")
                field_confidence['model_name'] = 0.6  # Lower confidence but don't reject

            elif any(generic in model.lower() for generic in ['tractor', 'vehicle']) and not any(c.isdigit() for c in model):
                # "Tractor" alone is too generic
                field_confidence['model_name'] = 0.5
            else:
                has_numbers = any(c.isdigit() for c in model)
                has_letters = any(c.isalpha() for c in model)
                # Good model names usually have numbers (855, 60, 745, etc.)
                if has_numbers and has_letters and len(model) > 3:
                    field_confidence['model_name'] = 0.95
                else:
                    field_confidence['model_name'] = 0.75

        
           # Horse power confidence
            hp = data.get('horse_power')
            if hp is None or hp == 0:
                field_confidence['horse_power'] = 0.0
            elif isinstance(hp, (int, float)) and 10 <= hp <= 200:  # Reasonable HP range
                field_confidence['horse_power'] = 0.95
            elif isinstance(hp, (int, float)):
                field_confidence['horse_power'] = 0.7  # Unusual HP value
            else:
                field_confidence['horse_power'] = 0.5  # Wrong type



        
            # Asset cost confidence
            cost = data.get('asset_cost')
            if cost is None or cost == 0:
                field_confidence['asset_cost'] = 0.0
            elif isinstance(cost, (int, float)) and 100000 <= cost <= 50000000:  # Reasonable cost range (1L-5Cr)
                field_confidence['asset_cost'] = 0.95
            elif isinstance(cost, (int, float)):
                field_confidence['asset_cost'] = 0.65  # Unusual cost
            else:
                field_confidence['asset_cost'] = 0.5  # Wrong type
        
            # Calculate average confidence
            avg_confidence = sum(field_confidence.values()) / len(field_confidence) if field_confidence else 0.0
        
            return {
                "dealer_name": data.get("dealer_name"),
                "model_name": data.get("model_name"),
                "horse_power": data.get("horse_power"),
                "asset_cost": data.get("asset_cost"),
                "confidence": round(avg_confidence, 2)
            }
        except Exception as e:
            print(f"  JSON parse error: {e}")
            print(f"  Raw response: {response_text}")
            return self._empty_result()

    def _ocr_fallback_hp(self, image):
        """Emergency OCR fallback to find HP if VLM fails"""
        import pytesseract
        import re
        from PIL import Image
        import cv2
    
        try:
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                img_pil = image
        
            # Get OCR text
            text = pytesseract.image_to_string(img_pil)
        
            # Search for HP patterns
            patterns = [
                r'(\d{2,3})\s*(?:HP|BHP|H\.P\.)',  # "48 HP", "50 BHP"
                r'(?:HP|BHP|H\.P\.)\s*[:\-]?\s*(\d{2,3})',  # "HP: 48", "HP-50"
            ]
        
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    hp = int(match.group(1))
                    if 15 <= hp <= 200:  # Reasonable range
                        print(f"  🔍 OCR found HP: {hp}")
                        return hp
        
            return None
        except:
            return None

    def _empty_result(self):
        """Return empty result when extraction fails"""
        return {
            "dealer_name": None,
            "model_name": None,
            "horse_power": None,
            "asset_cost": None,
            "confidence": 0.0
        }


# Alias for compatibility
NanonetsExtractor = Qwen2VLExtractor

