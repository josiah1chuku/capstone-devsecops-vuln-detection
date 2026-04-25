# Transformer-Driven Shift-Left Security
## Integrating Transformer Encoders into DevSecOps Pipelines

**Author:** Josiah Chuku  
**Institution:** Florida A&M University  
**Instructor:** Dr. Theran Carlos  
**Year:** 2026  

[![CI Pipeline](https://github.com/josiah1chuku/capstone-devsecops-vuln-detection/actions/workflows/evaluate.yml/badge.svg)](https://github.com/josiah1chuku/capstone-devsecops-vuln-detection/actions)

---

## Overview

VulnDetector is a hybrid deep learning model that detects vulnerabilities
in C/C++ source code by combining:

- **CodeBERT** - Pre-trained transformer for code token semantics
- **R-GCN** - Relational Graph Convolutional Network for Data Flow Graph analysis
- **Gated Fusion** - Adaptive combination of text and graph representations

The model integrates into a GitHub Actions CI/CD pipeline to enable
**shift-left security** - catching vulnerabilities at commit time.

---

## Results

Evaluated on DiverseVul test set (66,098 functions, 5.7% vulnerable):

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.7677** |
| Best Val F1 | **0.7794** |
| MCC | 0.2117 |
| Recall | 0.3774 |
| Precision | 0.1948 |
| Accuracy | 87.0% |

---

## Project Structure

```
capstone-devsecops-vuln-detection/
├── .github/workflows/
│   └── evaluate.yml       # CI/CD pipeline
├── step4_model/
│   └── full_model.py      # VulnDetector architecture
├── step5_train/
│   ├── prepare_data.py    # Build train/val/test splits
│   ├── build_dfg_cache.py # Extract Data Flow Graphs
│   └── train.py           # Training script
├── step6_eval/
│   └── evaluate.py        # Evaluation + plots
├── requirements.txt
└── README.md
```

---

## Setup and Installation

### Requirements
- Python 3.12
- NVIDIA GPU with 16GB+ VRAM (A100 recommended)
- Google Colab Pro or equivalent

### Step 1 - Clone the repository
```bash
git clone https://github.com/josiah1chuku/capstone-devsecops-vuln-detection.git
cd capstone-devsecops-vuln-detection
```

### Step 2 - Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 - Download dataset
Request DiverseVul from: https://github.com/wagner-group/diversevul  
Place the file at: `data/diversevul.json`

### Step 4 - Prepare data splits
```bash
python step5_train/prepare_data.py \
    --input_path data/diversevul.json \
    --output_dir data/
```

### Step 5 - Build Data Flow Graph cache
```bash
python step5_train/build_dfg_cache.py \
    --input_path  data/diversevul.json \
    --output_path data/dfg_cache.pkl
```

### Step 6 - Train the model
```bash
python step5_train/train.py \
    --train_path data/train_balanced_40k.csv \
    --val_path   data/val_balanced_4k.csv \
    --cache_path data/dfg_cache.pkl \
    --epochs 10 --batch_size 32 --lr 1e-4
```

Expected output:
```
Epoch  1/10 | Train Loss: 0.4483 F1: 0.7421 | Val F1: 0.7601 AUC: 0.8685
Epoch  2/10 | Train Loss: 0.3643 F1: 0.8205 | Val F1: 0.7794 AUC: 0.8733
Early stopping at epoch 6
Best checkpoint: checkpoints/best_model_final.pt
```

### Step 7 - Evaluate on test set
```bash
python step6_eval/evaluate.py \
    --checkpoint  checkpoints/best_model_final.pt \
    --test_path   data/test.csv \
    --cache_path  data/dfg_cache.pkl \
    --output_path results/eval_results.json \
    --mode full
```

Expected results:
```
AUC-ROC    : 0.7677
F1 Score   : 0.2570  (threshold = 0.7341)
MCC        : 0.2117
Precision  : 0.1948
Recall     : 0.3774
```

---

## Full Training Notebook (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17-HJnTymfBtEFXbRL1ZMkQtUEyzx-4EL)

---

## CI/CD Pipeline

The GitHub Actions workflow triggers on every push to `main` and every pull request:

1. Install dependencies
2. Run VulnDetector scan in CI mode
3. Upload results as artifact
4. Post metrics table as PR comment

---

## Dataset

**DiverseVul** (Chen et al., 2023)
- 330,486 C/C++ functions from 7,514 open-source projects
- 18,945 vulnerable functions across 150 CWE categories
- Source: https://github.com/wagner-group/diversevul

---

## License

MIT License