"""
Batch process all images in a folder
Much faster than running executable.py multiple times (model loaded once)
"""


import os
import sys
import glob
import json
import time
from pathlib import Path


# Add utils to path
sys.path.append(os.path.dirname(__file__))


# Import DocumentExtractor from executable
from executable import DocumentExtractor



def process_folder(input_folder, output_dir="./outputs", model_dir="./models"):
    """
    Process all images in a folder
    
    Args:
        input_folder: Folder containing images
        output_dir: Where to save results
        model_dir: Where models are stored
    """
    
    # Find all images (case-insensitive, no duplicates)
    all_files = list(Path(input_folder).iterdir())
    image_files = [str(f) for f in all_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print("="*60)
    
    # Initialize extractor ONCE (saves time!)
    print("\nInitializing pipeline (this happens only once)...")
    extractor = DocumentExtractor(model_dir=model_dir)
    
    # Process all images
    results_summary = []
    total_start = time.time()
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(image_files)}] Processing: {Path(image_path).name}")
        print(f"{'='*60}")
        
        try:
            result = extractor.process_single_image(image_path, output_dir)
            results_summary.append({
                "file": Path(image_path).name,
                "status": "SUCCESS",
                "time": result.get("processing_time_sec", 0),
                "confidence": result.get("confidence", 0),
                "fields_extracted": sum(1 for k, v in result.get("fields", {}).items() 
                                       if k not in ['signature', 'stamp'] and v not in [None, "", 0, False, []])
            })
        except Exception as e:
            print(f"ERROR processing {image_path}: {e}")
            results_summary.append({
                "file": Path(image_path).name,
                "status": "FAILED",
                "error": str(e)
            })
    
    # Print summary
    total_time = time.time() - total_start
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE!")
    print("="*60)
    
    successful = sum(1 for r in results_summary if r["status"] == "SUCCESS")
    failed = len(results_summary) - successful
    
    print(f"\nTotal images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(image_files)*100:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Avg time/image: {total_time/len(image_files):.2f}s")
    
    # Calculate average confidence
    avg_conf = sum(r.get('confidence', 0) for r in results_summary if r['status'] == 'SUCCESS') / max(successful, 1)
    print(f"Avg confidence: {avg_conf:.2f}")
    
    # Show per-image results
    print("\n" + "-"*60)
    print("PER-IMAGE RESULTS:")
    print("-"*60)
    for r in results_summary[:20]:  # Show first 20
        if r["status"] == "SUCCESS":
            print(f"✓ {r['file']:40s} | {r['time']:5.1f}s | Conf: {r['confidence']:.2f} | Fields: {r['fields_extracted']}/4")
        else:
            print(f"✗ {r['file']:40s} | ERROR: {r.get('error', 'Unknown')[:40]}")
    
    if len(results_summary) > 20:
        print(f"... and {len(results_summary) - 20} more")
    
    # Save summary
    summary_file = os.path.join(output_dir, "_batch_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "total_images": len(image_files),
            "successful": successful,
            "failed": failed,
            "total_time_sec": round(total_time, 2),
            "avg_time_per_image": round(total_time/len(image_files), 2),
            "avg_confidence": round(avg_conf, 2),
            "results": results_summary
        }, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_file}")
    print("="*60)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process images in a folder")
    parser.add_argument("input_folder", help="Folder containing images")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    parser.add_argument("--model-dir", default="./models", help="Model directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"Error: Folder not found: {args.input_folder}")
        sys.exit(1)
    
    process_folder(args.input_folder, args.output_dir, args.model_dir)
