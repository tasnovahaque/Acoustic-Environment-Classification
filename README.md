# Acoustic Environment Classification

This project implements an **Acoustic Environment Classification** system using a custom 1D CNN + Transformer architecture inspired by A-JEPA (Audio Joint Embedding Predictive Architecture). The code is provided in the notebook `iajepa-final-notebook (1).ipynb`.

## Project Overview

The goal is to classify audio recordings into different environment sound categories using deep learning. The model leverages:
- 1D CNN feature extraction (similar to wav2vec)
- Transformer encoder and decoder blocks
- Masking strategies for self-supervised pretraining
- Fine-tuning for supervised classification

## Dataset

- The dataset should be organized in class-named subfolders, each containing `.wav` files.
- Example path: `/kaggle/input/environment-sound/Environment_Sound`
- The notebook automatically detects classes from the dataset directory.

## Model Architecture

- **CNNFeatureExtractor**: Extracts features from raw audio.
- **AudioTransformer**: Transformer encoder for audio tokens.
- **TransformerDecoder**: Decoder for predictive learning.
- **AJEPA1D**: Combines context encoder, target encoder (EMA), predictor, and classifier head.

## Training Procedure

1. **Pretraining**: Self-supervised learning with curriculum masking strategies (block, sparse, contextual).
2. **Fine-tuning**: Supervised training for environment sound classification.
3. **Evaluation**: Accuracy and confusion matrix visualization.

## Usage

1. Update the `DATA_PATH` variable in the notebook to point to your dataset.
2. Run the notebook to train and evaluate the model.
3. Pretrained and fine-tuned model checkpoints are saved as `ajepa_1d_pretrained.pth` and `ajepa_1d_finetuned.pth`.
4. Confusion matrix is saved as `confusion_matrix.png`.

## Requirements

- Python
- PyTorch
- torchaudio
- scikit-learn
- einops
- tqdm
- matplotlib
- seaborn

Install required packages before running the notebook.

## Visualization

The notebook includes code to visualize masking strategies and learning curves.

## Reference

- Architecture inspired by [A-JEPA: Audio Joint Embedding Predictive Architecture].

---
For details, see the notebook: `iajepa-final-notebook (1).ipynb`