# Transparent Multi-Modal AI System 🔍🖼️

A BERT-ResNet fusion model with built-in explainability through attention visualization. Processes text-image pairs while exposing decision-making mechanisms via heatmaps and token highlighting.

## 🚀 Features
- **Dual Encoder Architecture**: BERT (text) + ResNet (image) with cross-attention fusion
- **Explainability Layer**: Real-time visualization of:
  - Image regions influencing decisions (OpenCV heatmaps)
  - Key text tokens (HTML/CSS highlighting)
  - Cross-modal attention patterns
- **CLIP-style Training**: Contrastive loss for alignment learning
- **Production Ready**: ONNX export & FastAPI deployment

## 📦 Installation
```
git clone https://github.com/ashworks1706/multimodal-llm
cd multimodal-llm
pip install -r requirements.txt
```

## 🛠️ Usage
1. **Data Preparation** (COCO dataset):
```
python data/preprocess_coco.py --images_dir /data/images/images --annotations /data/images/annotations
```

2. **Training**:
```
python training/train.py --batch_size 64 --lr 1e-5 --use_amp
```

3. **Interactive Demo**:
```
python demo/app.py
```

## 🧠 Project Structure
```
├── models/              # Encoders & fusion layers
├── data/                # COCO processing scripts
├── training/            # Contrastive loss implementation
├── explainability/      # Attention visualization tools
├── demo/                # Gradio interface
└── docs/                # Architecture diagrams & blog post
```

## 📐 Model Architecture

```mermaid
flowchart TB
    %% Main Flow with Integrated Blocks
    A[Data Collection & Preprocessing] --> B[Modality-Specific Encoders]
    B --> C[Multi-Modal Fusion Mechanism]
    C --> D[Training Pipeline]
    D --> E[Evaluation & Fine-Tuning]
    E --> F[Deployment & Documentation]

    %% Data Collection Blocks
    A1["Text Data"] --> A
    A2["Image Data"] --> A

    %% Encoder Blocks
    B1["Text Encoder (BERT)"] --> B
    B2["Image Encoder (ResNet)"] --> B

    %% Fusion Mechanism Blocks
    C1["Concatenation"] --> C
    C2["Cross-Attention"] --> C

    %% Training Pipeline Blocks
    D1["Contrastive Loss"] --> D
    D2["Hyperparameter Tuning"] --> D

    %% Evaluation Blocks
    E1["Quantitative Metrics"] --> E
    E2["Qualitative Analysis"] --> E

    %% Deployment Blocks
    F1["Model Export (ONNX)"] --> F
    F2["API Development (FastAPI)"] --> F
    F3["Documentation"] --> F


```

*Dual encoder system with cross-attention fusion and visualization hooks*

## 🏋️ Training Details
| Hyperparameter      | Value        |
|---------------------|--------------|
| Batch Size          | 64-128       |
| Learning Rate       | 1e-5 to 1e-4 |
| Embedding Dimension | 512          |
| Temperature (τ)     | 0.07         |

## 🎥 Demo Screenshot
![Coming Soon](docs/demo_preview.gif)

## 🤝 Contributing
PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---
