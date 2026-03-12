"""EDA for IDFC Invoice Dataset"""
import os, glob, json, cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sns.set_style("whitegrid")

def analyze_dataset():
    print("="*60)
    print("IDFC INVOICE DATASET - EDA")
    print("="*60)
    
    # Get images
    image_files = glob.glob("../training_data/*.jpg") + glob.glob("../training_data/*.png")
    print(f"\nTotal documents: {len(image_files)}")
    
    # Analyze sample
    file_sizes, dimensions = [], []
    for img_path in image_files[:50]:
        file_sizes.append(os.path.getsize(img_path) / 1024)
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            dimensions.append((w, h))
    
    # Get results
    result_files = glob.glob("../outputs/*_result.json")
    confidences, times = [], []
    field_success = defaultdict(int)
    
    for rf in result_files:
        with open(rf) as f:
            data = json.load(f)
            confidences.append(data.get('confidence', 0))
            times.append(data.get('processing_time_sec', 0))
            fields = data.get('fields', {})
            for field in ['dealer_name', 'model_name', 'horse_power', 'asset_cost']:
                if fields.get(field):
                    field_success[field] += 1
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('IDFC Invoice Dataset - EDA', fontsize=16, fontweight='bold')
    
    # File sizes
    axes[0,0].hist(file_sizes, bins=20, color='skyblue', edgecolor='black')
    axes[0,0].set_title('File Size Distribution')
    axes[0,0].set_xlabel('Size (KB)')
    axes[0,0].axvline(np.mean(file_sizes), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(file_sizes):.0f}KB')
    axes[0,0].legend()
    
    # Dimensions
    if dimensions:
        w, h = zip(*dimensions)
        axes[0,1].scatter(w, h, alpha=0.6, c='green')
        axes[0,1].set_title('Image Dimensions')
        axes[0,1].set_xlabel('Width (px)')
        axes[0,1].set_ylabel('Height (px)')
    
    # Languages (heuristic)
    lang_dist = {"English": 0, "Hindi": 0, "Mixed": 0}
    for f in image_files:
        if any(x in f.lower() for x in ['hin', 'hindi']): lang_dist["Hindi"] += 1
        elif any(x in f.lower() for x in ['eng', 'english']): lang_dist["English"] += 1
        else: lang_dist["Mixed"] += 1
    
    axes[0,2].bar(lang_dist.keys(), lang_dist.values(), color=['#ff9999','#66b3ff','#99ff99'])
    axes[0,2].set_title('Language Distribution (Heuristic)')
    for i, (k,v) in enumerate(lang_dist.items()):
        axes[0,2].text(i, v+2, str(v), ha='center', fontweight='bold')
    
    # Confidence
    if confidences:
        axes[1,0].hist(confidences, bins=15, color='coral', edgecolor='black')
        axes[1,0].set_title('Confidence Distribution')
        axes[1,0].set_xlabel('Confidence Score')
        axes[1,0].axvline(np.mean(confidences), color='red', linestyle='--',
                         label=f'Mean: {np.mean(confidences):.2f}')
        axes[1,0].legend()
    
    # Processing time
    if times:
        axes[1,1].hist(times, bins=15, color='lightgreen', edgecolor='black')
        axes[1,1].set_title('Processing Time Distribution')
        axes[1,1].set_xlabel('Time (seconds)')
        axes[1,1].axvline(np.mean(times), color='red', linestyle='--',
                         label=f'Mean: {np.mean(times):.1f}s')
        axes[1,1].legend()
    
    # Field success
    if field_success and result_files:
        fields = list(field_success.keys())
        rates = [field_success[f]/len(result_files)*100 for f in fields]
        bars = axes[1,2].barh(fields, rates, color='purple')
        axes[1,2].set_title('Field Extraction Success Rate')
        axes[1,2].set_xlabel('Success Rate (%)')
        for bar, rate in zip(bars, rates):
            axes[1,2].text(rate+1, bar.get_y()+bar.get_height()/2, 
                          f'{rate:.0f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: eda_analysis.png")
    
    # Summary
    summary = {
        "total_documents": len(image_files),
        "processed": len(result_files),
        "avg_confidence": round(np.mean(confidences), 2) if confidences else 0,
        "avg_time_sec": round(np.mean(times), 1) if times else 0,
        "field_success_rates": {k: f"{v/len(result_files)*100:.0f}%" 
                                for k,v in field_success.items()} if result_files else {}
    }
    with open('eda_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ Saved: eda_summary.json")
    print("="*60)

if __name__ == "__main__":
    analyze_dataset()
