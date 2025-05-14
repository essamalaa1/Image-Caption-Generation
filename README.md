# Image Captioning with Flickr8k, ResNet50, and GRU

This project implements an image captioning model using the Flickr8k dataset. It employs a pre-trained ResNet50 model as the image feature extractor (encoder) and a GRU-based network as the caption generator (decoder). Two variants are explored:

1. **Not Fine-Tuned**: ResNet50 encoder’s weights are frozen.  
2. **Fine-Tuned**: Layers from `conv5_block1_out` onward in ResNet50 are unfrozen and jointly trained with the decoder.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Dataset & Pre-trained Embeddings](#dataset--pre-trained-embeddings)  
4. [Model Architecture](#model-architecture)  
5. [Setup & Installation](#setup--installation)  
   - [Prerequisites](#prerequisites)  
   - [Dependencies](#dependencies)  
   - [Data Download & Preparation](#data-download--preparation)  
6. [Usage](#usage)  
7. [Code Structure Highlights](#code-structure-highlights)  
8. [Evaluation Metrics](#evaluation-metrics)  
9. [Results Summary](#results-summary)  
10. [Sample Outputs](#sample-outputs)  
11. [Potential Future Work](#potential-future-work)  

---

## Project Overview

Automatically generate textual captions for images using an encoder–decoder framework:

- **Encoder**: Pre-trained ResNet50 (ImageNet) → feature maps → GAP → Dense → image context vector.  
- **Decoder**: GRU network with GloVe embeddings, conditioned on image context to predict next word.

---

## Features

- Caption preprocessing: tokenization, padding, `<start>`/`<end>` tokens.  
- Image resizing & preprocessing for ResNet50.  
- GloVe (6B.100d) word embeddings integration.  
- Dual-model training (frozen vs. fine-tuned encoder).  
- Evaluation: BLEU, perplexity (loss), token-level accuracy, cosine similarity.  
- Visualization: cosine similarity distributions, side-by-side caption comparisons.

---

## Dataset & Pre-trained Embeddings

- **Flickr8k Dataset**  
  - 8,000 images with 5 captions each.  
  - Captions file: `Flickr8k.token.txt`.  
  - Images archive: `Flickr8k_Dataset.zip`.  

- **GloVe 6B (100d)**  
  - 100-dim vectors over 6B tokens: `glove.6B.100d.txt`.

---

## Model Architecture

1. **Image Encoder (ResNet50)**  
   - Input: `(224, 224, 3)`  
   - `include_top=False`, GlobalAveragePooling2D  
   - Dense → image context vector (e.g., 1024 dims)

2. **Caption Decoder (GRU)**  
   - Embedding layer (initialized with GloVe, `mask_zero=True`)  
   - GRU (`units=1024`, `return_sequences=True`)  
   - Dropout for regularization  
   - TimeDistributed Dense (1024) → TimeDistributed Dense (vocab size, `softmax`)

---

## Setup & Installation

### Prerequisites

- Python 3.7+  
- `pip`  
- (Optional) NVIDIA GPU with CUDA & cuDNN

### Dependencies

Create a `requirements.txt`:

```text
numpy
pandas
tqdm
matplotlib
gensim
Pillow
scikit-learn
nltk
tensorflow>=2.6.0      # or tensorflow-gpu

pip install -r requirements.txt

# Images
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
unzip Flickr8k_Dataset.zip -d Flickr8k_Dataset/

# Captions
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
unzip Flickr8k_text.zip -d Flickr8k_text/

# GloVe embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
