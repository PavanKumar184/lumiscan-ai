# Kaggle Training Guide

## Dataset

Use this Kaggle dataset:

https://www.kaggle.com/datasets/nih-chest-xrays/data

## File to Use

Use:

`kaggle_train_nih_dr_ultra.py`

This is a notebook-style Python script with cell markers. You can copy its content into a Kaggle notebook cell-by-cell or paste the full script into one Kaggle notebook.

Important project framing:

DR-Ultra is the main proposed framework. DenseNet121, ResNet50, and EfficientNet-B0 are not the main model by themselves. They are used as selectable visual encoder backbones inside the DR-Ultra training pipeline.

## Kaggle Setup Steps

1. Open Kaggle.
2. Create a new notebook.
3. Click `Add Input`.
4. Search for `NIH Chest X-rays`.
5. Add the dataset with slug:
   `nih-chest-xrays/data`
6. Set accelerator:
   `GPU T4 x2` if available, otherwise `GPU T4`.
7. Copy the content of `kaggle_train_nih_dr_ultra.py` into the notebook.
8. Run all cells.

## Important Training Settings

Inside the notebook, you can tune:

```python
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-4
VISUAL_BACKBONE = "densenet121"
```

Recommended for better results:

```python
EPOCHS = 10
VISUAL_BACKBONE = "densenet121"
```

If Kaggle GPU memory is low:

```python
BATCH_SIZE = 16
```

## Output Files

The notebook saves outputs in:

`/kaggle/working/dr_ultra_outputs`

Important files:

- `best_dr_ultra_densenet121_nih.pth`
- `class_weights.csv`
- `training_history.csv`
- `test_metrics.json`
- `inference_config.json`
- `classes.json`

It also creates:

`/kaggle/working/dr_ultra_trained_model_package.zip`

Download this zip file from the Kaggle notebook output panel.

## What the Model Checkpoint Contains

The `.pth` file contains:

- framework name: `DR-Ultra`
- selected visual backbone name
- model architecture name
- trained model weights
- class names
- image size
- prediction threshold
- class imbalance weights
- best validation metrics

## Using the Model in Your Project

Later, copy the downloaded `.pth` file into your project folder and load it using the same architecture used during training.

The model outputs 14 logits. Apply sigmoid:

```python
probabilities = torch.sigmoid(logits)
```

Each probability corresponds to one disease class.

## Important Note

This notebook trains the DR-Ultra image branch using a selectable visual backbone and imbalance-aware weighted loss. The full vision-language DR-Ultra model can be built on top of this by adding a text encoder and fusion module later. This model is a practical trainable DR-Ultra-stage checkpoint that fits your current project stage and can be used in the application.
