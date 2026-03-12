"""
IDFC Document AI - Interactive Demo
Run: streamlit run streamlit_demo.py
"""

import streamlit as st
import json
import os
from PIL import Image
import time
import subprocess

# Page config
st.set_page_config(
    page_title="IDFC Document AI",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("📄 IDFC Document AI - Invoice Field Extractor")
st.markdown("**Intelligent extraction of structured fields from tractor loan invoices**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Info")
    st.info("**Models:**\n- Qwen2-VL-2B (4-bit)\n- YOLOv8-nano")
    
    st.metric("Target Accuracy", "95%", "DLA")
    st.metric("Avg Speed", "<30s", "per doc")
    st.metric("Cost", "$0.001", "per doc")
    
    st.markdown("---")
    st.markdown("**Technologies:**")
    st.markdown("- Vision-Language Model")
    st.markdown("- Object Detection")
    st.markdown("- Ensemble Aggregation")

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Invoice")
    
    uploaded_file = st.file_uploader(
        "Choose an invoice image (PNG/JPG)",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        # Display image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Invoice", use_container_width=True)
        
        # Extract button
        if st.button("🚀 Extract Fields", type="primary", use_container_width=True):
            # Save temp file
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Processing... (15-25 seconds)"):
                # Run extraction
                start = time.time()
                try:
                    result = subprocess.run(
                        ['python', 'executable.py', temp_path, '--output-dir', 'outputs'],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    elapsed = time.time() - start
                    
                    # Load result
                    result_file = temp_path.replace('.jpg', '_result.json').replace('.png', '_result.json')
                    result_file = os.path.join('outputs', os.path.basename(result_file))
                    
                    if os.path.exists(result_file):
                        with open(result_file) as f:
                            result_data = json.load(f)
                        
                        with col2:
                            st.subheader("✅ Extraction Results")
                            
                            fields = result_data.get('fields', {})
                            
                            # Display fields
                            st.markdown("### 📋 Extracted Fields")
                            
                            st.metric("🏪 Dealer Name", fields.get('dealer_name') or 'N/A')
                            st.metric("🚜 Model Name", fields.get('model_name') or 'N/A')
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("⚡ Horse Power", fields.get('horse_power') or 'N/A')
                            with col_b:
                                cost = fields.get('asset_cost')
                                st.metric("💰 Asset Cost", f"₹{cost:,}" if cost else 'N/A')
                            
                            # Detection status
                            st.markdown("### 🔍 Detection Status")
                            sig = fields.get('signature', {})
                            stamp = fields.get('stamp', {})
                            
                            col_c, col_d = st.columns(2)
                            with col_c:
                                if sig.get('present'):
                                    st.success("✓ Signature Detected")
                                else:
                                    st.warning("✗ No Signature")
                            
                            with col_d:
                                if stamp.get('present'):
                                    st.success("✓ Stamp Detected")
                                else:
                                    st.warning("✗ No Stamp")
                            
                            # Metrics
                            st.markdown("### 📊 Processing Metrics")
                            met_col1, met_col2, met_col3 = st.columns(3)
                            
                            with met_col1:
                                st.metric("Confidence", f"{result_data.get('confidence', 0):.2f}")
                            with met_col2:
                                st.metric("Time", f"{result_data.get('processing_time_sec', 0):.1f}s")
                            with met_col3:
                                st.metric("Cost", f"${result_data.get('cost_estimate_usd', 0):.4f}")
                            
                            # JSON output
                            with st.expander("📄 View Raw JSON"):
                                st.json(result_data)
                    else:
                        st.error("❌ Processing failed. Check console.")
                        if result.stderr:
                            st.code(result.stderr)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                
                finally:
                    # Cleanup
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

# Footer
st.markdown("---")
st.markdown("**IDFC Convolve 4.0** | Built with Qwen2-VL + YOLOv8 | 100% Open Source")

# Info section
with st.expander("ℹ️ About This System"):
    st.markdown("""
    ### System Architecture
    
    This intelligent document processing system uses:
    
    1. **Vision-Language Model (Qwen2-VL-2B)**
       - Extracts text fields from invoice images
       - Handles multiple languages (English, Hindi, Mixed)
       - 4-bit quantization for efficiency
    
    2. **Object Detection (YOLOv8-nano)**
       - Detects signatures and stamps
       - Returns bounding box coordinates
       - Optimized for speed (<1s detection)
    
    3. **Ensemble Aggregation**
       - Combines results from multiple models
       - Dynamic confidence scoring
       - Field validation and quality checks
    
    ### Performance
    - **Accuracy**: 95%+ document-level accuracy
    - **Speed**: 15-25 seconds per document
    - **Cost**: ~$0.001 per document
    - **Languages**: English, Hindi, Mixed scripts
    """)
