# Word-Level Stress Detection Using Speech-Text Unification

This repository contains the code and data structure for our project on detecting word-level stress in English speech using both speech-only and multimodal (speech + text) approaches. We explore various classifier heads built on top of pretrained `wav2vec 2.0` representations and propose a cross-attention fusion model integrating BERT-based text embeddings for improved performance.

> 📚 This project was conducted as part of the **IASNLP Summer School 2025**.

---

## 📁 Project Structure

```text
.
├── GER/                      # Contains speech audio files (.wav) from German L2 English speakers
│   ├── train/                # Training set audio files
│   └── test/                 # Test set audio files
├── labels/                   # CSV files with word-level stress labels for training
├── wav2vec2_embeddings/      # Extracted wav2vec2 frame-level embeddings for each .wav file
└── word/                     # .mat files with transcript + word alignment (start-end timings)
```

---

## 🧠 Overview

Word-level stress is a key prosodic feature affecting fluency, intelligibility, and meaning in spoken English. Misplaced stress is a common issue among non-native speakers, motivating the need for automatic detection systems.

### Our Contributions

- Extracted word-wise embeddings from frozen `wav2vec 2.0` pretrained models using word-aligned audio.
- Trained multiple speech-based classifiers (MLP, LSTM, Attention, Transformer) on these embeddings.
- Proposed a speech-text fusion model using `BERT` and `wav2vec2`, integrated via a cross-attention module.
- Achieved notable improvements in stress classification with multimodal fusion over pure speech-based models.

---

## 🏗️ Methodology

1. **Embedding Extraction**  
   Raw audio is processed using pretrained wav2vec2. Using word boundaries from `.mat` alignment files, frame-level embeddings are aggregated to obtain fixed-size word-level embeddings.

2. **Speech-based Classifiers**  
   We experimented with different classification heads:  
   - MLP  
   - LSTM  
   - Attention mechanism  
   - Transformer encoder

3. **Multimodal Fusion Classifier**  
   - BERT embeddings are extracted for transcribed words.
   - Cross-attention is used to fuse speech and text modalities.
   - Final embeddings are passed through a linear layer for binary classification (stressed / not stressed).

---

## 📊 Results Summary

| Model           | Accuracy | F1 Score |
|----------------|----------|----------|
| MLP (Speech)    | 76.82%   | 73.54%   |
| LSTM (Speech)   | 76.09%   | 72.50%   |
| Transformer     | 73.85%   | 65.18%   |
| Attention Head  | 74.49%   | 65.74%   |
| **Fusion (BERT + wav2vec2)** | **82.51%** | **82.02%** |

---

## 📦 Dependencies

- Python 3.8+
- PyTorch
- Transformers (`huggingface`)
- scipy, numpy, pandas, scikit-learn
- torchaudio

---

## 📂 Data Sources

- ISLE Corpus (German subset): Contains English speech from non-native German speakers.
- Word alignment files provided in `.mat` format with start-end times.

---

## ✍️ Authors

- Rishab Sharma  
- Pratyoosh Sharma  
- Raviteja Kamarajugadda

