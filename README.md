# Thyroid Cancer Detection — Hybrid Models

Hybrid deep learning models for **thyroid nodule classification** from ultrasound images, classifying each image into one of three categories: **normal**, **benign**, or **malignant**. Both models use a pretrained EfficientNet CNN as the image feature extractor, paired with two different classification heads.

This repository contains two experiments on the same dataset:

| Notebook | Feature extractor | Classifier head |
|----------|-------------------|-----------------|
| `Thyroid_Cancer_Detection_on_image_Dataset.ipynb` | EfficientNet-B4 | Graph Attention Network (GAT) |
| `Hybrid_CNN+SVM_for_thyroid_cancer_detection.ipynb` | EfficientNet-B3 | Support Vector Machine (RBF) |

> **Note on scope:** These are research/experimental notebooks and are **not** intended for clinical use or real diagnosis.

## Approach

**CNN + GAT model**
1. EfficientNet-B4 (ImageNet weights) extracts a 1792-dim feature vector per 224×224 image, projected to 512 dims.
2. Features become graph nodes; each node connects to near neighbours within a batch, and two stacked multi-head `GATConv` layers (4 heads each) refine them via attention.
3. A final linear layer maps to the 3 classes. Trained end-to-end with augmentation, AdamW, a StepLR schedule, and gradient clipping, checkpointing the best model by F1.

**CNN + SVM model**
1. EfficientNet-B3 (ImageNet weights) is used as a frozen feature extractor — the classifier head is removed and features are extracted without fine-tuning.
2. An RBF-kernel SVM is trained on the extracted features to predict the 3 classes.

## Results

Best model on the held-out test split:

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| EfficientNet-B4 + GAT | 0.922 | 0.922 | 0.922 | 0.922 |
| EfficientNet-B3 + SVM | 0.836 | 0.910 | 0.836 | 0.853 |

> Figures come from single runs; GAT test metrics in particular fluctuate across epochs, so the reported numbers are from the best checkpoint. Treat them as indicative rather than definitive.

## Getting started

### Requirements

Developed in Google Colab with a GPU (T4) runtime. Main dependencies:

```
torch
torchvision
torch_geometric
scikit-learn
numpy
pillow
kagglehub
```

Install with:

```bash
pip install -r requirements.txt
```

### Data

The dataset is **not** included in this repository due to its size (~103 MB). Download it from Kaggle:

- **AUITD (Algeria Ultrasound Images Thyroid Dataset):** `azouzmaroua/algeria-ultrasound-images-thyroid-dataset-auitd`

Both notebooks download this automatically via the Kaggle API. You'll need your own Kaggle credentials configured. The expected layout after extraction is `train/<class>/*.jpg` and `test/<class>/*.jpg`.

### Running

Open either notebook in Google Colab (or Jupyter with a GPU) and run the cells top to bottom. Adjust the dataset paths if you're not on Colab.

## Repository structure

```
.
├── Thyroid_Cancer_Detection_on_image_Dataset.ipynb   # EfficientNet-B4 + GAT
├── Hybrid_CNN+SVM_for_thyroid_cancer_detection.ipynb # EfficientNet-B3 + SVM
├── requirements.txt
├── .gitignore
└── README.md
```

## License

Add a license of your choice (e.g. MIT) if you want others to reuse this work. The underlying dataset has its own license and usage terms — check Kaggle before redistributing any data.
