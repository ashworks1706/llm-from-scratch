# Transparent Multi-Modal AI System 🔍🖼️

A BERT-ResNet fusion model with built-in explainability through attention visualization. Processes text-image pairs while exposing decision-making mechanisms via heatmaps and token highlighting.

**Begin [HERE](https://github.com/ashworks1706/ExplainableAI/blob/main/tutorial.ipynb)!**

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
├── data/                # Processing scripts
├── training/            # Contrastive loss implementation
├── explainability/      # Attention visualization tools
├── demo/                # Gradio interface
```

## 📐 Model Architecture

### Text Modal LLM

```mermaid
flowchart TB
  %% Main Flow
  A[Data Collection & Preprocessing] --> B[Text Encoder]
  B --> C[Training Pipeline]
  C --> D[Evaluation & Fine-Tuning]
  D --> E[Deployment & Documentation]

  %% Data Collection Blocks
  A1["Text Corpus"] --> A
  A2["Tokenization"] --> A

  %% Encoder Blocks
  B1["BERT Architecture"] --> B
  B2["Attention Mechanisms"] --> B

  %% Training Pipeline Blocks
  C1["Masked Language Modeling"] --> C
  C2["Hyperparameter Tuning"] --> C

  %% Evaluation Blocks
  D1["Quantitative Metrics"] --> D
  D2["Qualitative Analysis"] --> D

  %% Deployment Blocks
  E1["Model Export (ONNX)"] --> E
  E2["API Development (FastAPI)"] --> E
  E3["Documentation"] --> E
```

*Single encoder system with self-attention mechanisms and explainability features*

### Multimodal LLM

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
| ------------------- | ------------ |
| Batch Size          | 64-128       |
| Learning Rate       | 1e-5 to 1e-4 |
| Embedding Dimension | 512          |
| Temperature (τ)    | 0.07         |

---

### TODO

* [X] Decide the project structure and data to use
* [X] Setup repository, project and environment
* [ ] Phase 1

  * [X] Perform data analysis on textual data
  * [ ] Layout the key foundational points of modern transformer model and their difference from older models, provide examples
  * [ ] Perform different techniques in the architecture with test data
  * [ ] Train the model
  * [ ] Evaluate the model and explain why's of everything
  * [ ] Research and perform better techniques on the model while documenting
  * [ ] visualize the model attention via custom UI libraries
  * [ ] Finsh off with text model and save best to models
* [ ] Phase 2

  * [ ] Introduction to multimodal, visualize model
  * [ ] Perform data analysis on image-text pair data
  * [ ] Layout the key foundational points of modern transformer model and their difference from older models, provide examples
  * [ ] Perform different techniques in the architecture with test data
  * [ ] Train the model
  * [ ] Evaluate the model and explain why's of everything
  * [ ] Research and perform better techniques on the model while documenting
  * [ ] visualize the model attention via custom UI libraries
  * [ ] Finsh off with text model and save best to models
* [ ] Final edits, errors and notes
* [ ] DONE!

## 💎 Collaborators

Feel free to reach out on [discord](https://discord.gg/u6Gv4Rvr)!
