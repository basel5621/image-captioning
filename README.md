#  Image Captioning with Attention Mechanism

This project implements an image captioning model using a CNN-based encoder and an LSTM-based decoder with attention. The model generates descriptive captions for input images, trained on the MS COCO dataset.

---

## Overview

We use a pre-trained **MobileNetV3Large** as a feature extractor (encoder) and an **LSTM network with attention** as the decoder to generate image captions. The attention mechanism helps the decoder focus on specific parts of the image while predicting each word in the caption.

---

## Dataset

- **Original Dataset Used in Paper**: MS COCO 2014 (not available)
- **Dataset Used**: MS COCO 2017 from Kaggle  
  - Due to resource limitations, we trained the model on a **subset of 100,000 images**.
  - Captions were tokenized and padded; images resized to 224×224.

---

## Model Architecture

### Encoder
- **MobileNetV3Large** (`pretrained on ImageNet`)
- `include_top=False`, `include_preprocessing=True`
- Extracted features reshaped to sequences for attention mechanism

### Attention
- Custom attention layer using Dense + Add + Softmax layers
- Generates context vector for each LSTM timestep

### Decoder
- Token Embedding Layer
- LSTM with initial state derived from average CNN features
- Dense output layer with softmax over vocabulary

---

## Training

- **Loss Function**: Custom masked cross-entropy (ignores padding)
- **Optimizer**: Adam (`learning_rate=4e-4`)
- **Evaluation**: Custom BLEU-4 score callback on validation set

---

## Limitations

- Could not reproduce the exact results from the paper:
  - **Different dataset**: COCO 2017 vs COCO 2014
  - **Limited training**: Only 100k samples due to 12-hour runtime limit on Kaggle GPU
  - **Fewer epochs** due to time/memory constraints

