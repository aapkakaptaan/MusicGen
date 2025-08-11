# 🎵 LSTM Music/Sequence Generation with PyTorch & Comet ML

This project implements a **character-level LSTM** to learn patterns from a music dataset and generate new sequences (songs) character by character.  
It uses **PyTorch** for model building and training, and **Comet ML** for tracking experiments and results.  
The dataset and preprocessing utilities come from the [`mitdeeplearning`](https://github.com/lexfridman/mit-deep-learning) package.

---

## 📌 Features
- **Character-level sequence modeling** with `LSTMModel`
- Preprocessing with:
  - `vectorize_string()` for character → index mapping
  - `get_batch()` for training mini-batches
- Model training with `train_step()` and experiment logging via **Comet ML**
- Text generation with `generate_text()` using a trained LSTM model
- GPU acceleration (if available)

---

## 📦 Requirements
Install dependencies:
```bash
pip install torch comet_ml mitdeeplearning
