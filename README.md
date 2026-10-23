# Thyroid Cancer Detection — CNN + GAT

A hybrid deep learning model for **thyroid nodule classification** from ultrasound images. It combines an **EfficientNet-B4** convolutional feature extractor with a **Graph Attention Network (GAT)** head, classifying each image into one of three categories: **normal**, **benign**, or **malignant**.

> **Note on scope:** This is a research/experimental notebook and is **not** intended for clinical use or real diagnosis.

## Approach

1. **Image branch** — a pretrained EfficientNet-B4 (ImageNet weights) extracts a 1792-dimensional feature vector from each 224×224 ultrasound image, projected down to 512 dimensions.
2. **Graph branch** — image features become nodes in a graph. A lightweight edge-construction scheme connects each node to its near neighbours within a batch, and two stacked multi-head `GATConv` layers (4 heads each) refine the representations through attention.
3. **Classifier** — a final linear layer maps the attended features to the 3 output classes.

Training uses data augmentation (rotation, flips, color jitter, random grayscale), AdamW with weight decay, a StepLR schedule, and gradient clipping. The best model by F1 is checkpointed during training and reloaded for final evaluation.

## Results

Best model on the held-out test split:

| Metric | Score |
|--------|-------|
| Accuracy | 0.922 |
| Precision | 0.922 |
| Recall | 0.922 |
| F1 | 0.922 |

> Test metrics fluctuate considerably across epochs; the reported figures are from the best checkpoint. Treat them as indicative of a single run rather than a definitive benchmark.

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

The notebook downloads this automatically via the Kaggle API. You'll need to configure your own Kaggle credentials. The expected layout after extraction is `train/<class>/*.jpg` and `test/<class>/*.jpg`.

### Running

Open the notebook in Google Colab (or Jupyter with a GPU) and run the cells top to bottom. Adjust the dataset paths if you're not on Colab.

## Repository structure

```
.
├── Thyroid_Cancer_Detection_on_image_Dataset.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## License

Add a license of your choice (e.g. MIT) if you want others to reuse this work. The underlying dataset has its own license and usage terms — check Kaggle before redistributing any data.
