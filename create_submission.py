"""Create submission ZIP package (code only, no model weights)"""

import os
import shutil
import glob
from datetime import datetime

def create_submission_zip():
    """Package everything for submission"""
    
    print("="*60)
    print("Creating IDFC Hackathon Submission Package")
    print("="*60)
    
    # Files to include
    files_to_include = [
        "executable.py",
        "batch_process.py",
        "streamlit_demo.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        "yolov8n.pt",  # Include YOLO model (only 6MB)
        "utils/__init__.py",
        "utils/pdf_processor.py",
        "utils/vlm_extractor.py",
        "utils/yolo_detector.py",
        "utils/ensemble.py",
    ]
    
    # Optional analysis files
    optional_files = [
        "analysis/eda.py",
        "analysis/eda_analysis.png",
        "analysis/eda_summary.json"
    ]
    
    # Create temp directory
    temp_dir = "submission_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy main files
    print("\nCopying files...")
    for file_path in files_to_include:
        if os.path.exists(file_path):
            dest = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(file_path, dest)
            print(f"  ✓ {file_path}")
        else:
            print(f"  ⚠ Missing: {file_path}")
    
    # Copy optional files
    for file_path in optional_files:
        if os.path.exists(file_path):
            dest = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(file_path, dest)
            print(f"  ✓ {file_path}")
    
    # Copy 3 sample outputs
    sample_outputs = glob.glob("outputs/*_result.json")[:3]
    if sample_outputs:
        os.makedirs(os.path.join(temp_dir, "sample_output"), exist_ok=True)
        for output_file in sample_outputs:
            dest = os.path.join(temp_dir, "sample_output", os.path.basename(output_file))
            shutil.copy2(output_file, dest)
        print(f"  ✓ {len(sample_outputs)} sample outputs")
    
    # Create ZIP
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f"IDFC_DocumentAI_Submission_{timestamp}"
    
    print(f"\nCreating ZIP file...")
    shutil.make_archive(zip_name, 'zip', temp_dir)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    # Summary
    zip_size_mb = os.path.getsize(zip_name + '.zip') / 1024 / 1024
    
    print("\n" + "="*60)
    print("✅ SUBMISSION PACKAGE CREATED!")
    print("="*60)
    print(f"\nFile: {zip_name}.zip")
    print(f"Size: {zip_size_mb:.1f} MB")
    print(f"\n📁 Contents:")
    print(f"  • Python source code")
    print(f"  • Requirements & README")
    print(f"  • YOLOv8 model (6MB)")
    print(f"  • Analysis & visualizations")
    print(f"  • Sample outputs")
    print(f"\n⚠️  NOT included (will auto-download):")
    print(f"  • models/ folder (10GB - auto-downloads on first run)")
    print(f"\n📤 Ready to upload to Google Drive!")
    print(f"   Estimated upload time: ~1-2 minutes")
    print("="*60)

if __name__ == "__main__":
    create_submission_zip()
