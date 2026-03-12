# IDFC Document AI - Invoice Field Extraction

**Hackathon**: IDFC Convolve 4.0  
**Problem Statement**: Intelligent Document AI for Tractor Loan Invoice Processing  



## Solution Overview

Automated field extraction from tractor loan invoices using a multi-modal ensemble approach combining Vision-Language Models and object detection.
## System Architecture
INPUT DOCUMENT (PDF/Image) 
PDF PROCESSOR 
(pypdfium2)

QWEN2-VL YOLOv8-nano 
(4-bit)  Detection
Extracts: Detects:
- Dealer - Signature 
- Model - Stamp 
- HP - Bboxes
- Cost 
     ▼
 ENSEMBLE 
 AGGREGATOR 
 - Combines
 - Validates
 - Scores
 JSON OUTPUT 
 + Confidence 


### Installation

```bash
# Extract submission
unzip IDFC_DocumentAI_Submission.zip
cd idfc_doc_extractor

# Create environment
conda create -n idfc python=3.10
conda activate idfc

# Install dependencies
pip install -r requirements.txt


# Models auto-download on first run (~10GB, one-time, 10-20 min)
