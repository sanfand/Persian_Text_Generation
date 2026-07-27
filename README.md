# Persian Tweet Text Generation

A comparative study of four sequence models for Persian text generation, trained on a large dataset of Persian tweets.

## Models

| Model | Architecture | Parameters |
|-------|-------------|------------|
| LSTM | 2-layer LSTM + LayerNorm | ~7M |
| Transformer | GPT-style Decoder (4 layers, 8 heads) | ~18M |
| CNN | Dilated Causal CNN (4 blocks, dilation 1-2-4-8) | ~12M |
| SVM | TF-IDF + LinearSVC (top-3000 tokens) | — |

All models share the same tokenizer: [`HooshvareLab/gpt2-fa`](https://huggingface.co/HooshvareLab/gpt2-fa) (~42k BPE vocab).

---

## Project Structure

```
├── 01_svm.ipynb                    # SVM training notebook
├── 02_cnn.ipynb                    # Dilated Causal CNN training notebook
├── 03_lstm.ipynb                   # LSTM training notebook
├── 04_transformer.ipynb            # GPT-style Transformer training notebook
└── inference.ipynb      # Run all 4 models — no training required
```

---

## Quick Start — Inference Only

If you just want to generate text with the trained models (no training needed):

1. Download `Persian_TG_Inference.ipynb`
2. Upload your trained model files to Google Drive:

```
MyDrive/Persian_Text_Generation/
├── models/lstm/
│   ├── model.pt
│   └── tokenizer/
├── models/transformer/
│   ├── model.pt
│   └── tokenizer/
├── models/cnn/
│   ├── model.pt
│   └── tokenizer/
└── /models/svm/
    ├── svm_pipeline.joblib
    └── svm_meta.json
```

3. Open the notebook in Google Colab
4. Set your seed word in **cell 2**:
```python
SEED_TEXT = 'امروز'   # change to any Persian word or phrase
```
5. Run all cells — all four models generate text from your prompt

---

## Training From Scratch

### Requirements

- Google Colab (GPU runtime recommended — T4 or better)
- Google Drive with the dataset

### Dataset

> **Note:** This project was trained on a private dataset of Persian tweets that is not publicly available and cannot be shared.

If you want to train the models yourself, you will need to provide your own Persian text dataset as a `.csv` file and update the `DRIVE_PATH` variable in the config cell of each notebook accordingly.

### Running

Open any training notebook in Colab and run all cells in order.  
Each notebook handles extraction, preprocessing, tokenization, training, and generation automatically.

**Estimated training time on T4 GPU:**

| Model | Time per Epoch | Total (10 epochs) |
|-------|---------------|-------------------|
| LSTM | 8–10 min | ~90 min |
| Transformer | 5–7 min | ~65 min |
| CNN | 3–5 min | ~45 min |
| SVM | — (batch) | ~60–90 min (CPU) |

> **Note:** SVM does not use the GPU. Training runs entirely on CPU.

### Checkpoint Resume

All deep learning notebooks (LSTM, Transformer, CNN) save a checkpoint after every epoch to Google Drive and automatically delete the previous one to save space.  
If your Colab session disconnects, simply re-run the training cell — it will resume from the last completed epoch.

---

## Generation

All models use the following decoding strategy at inference time:

- **Nucleus (top-p) sampling** — keeps the smallest set of tokens whose cumulative probability exceeds `top_p`
- **Temperature scaling** — controls output diversity
- **Repetition penalty** — reduces the likelihood of repeating already-generated tokens

Default parameters (adjustable in each notebook):

```python
TEMPERATURE  = 0.9
TOP_P        = 0.9
REP_PENALTY  = 1.5
MAX_NEW_TOKENS = 25
```

---

## Results

Example outputs for the seed word **"امروز"** (today):

| Model | Generated Text |
|-------|---------------|
| LSTM | امروز صبح با دوستم ... |
| Transformer | امروز هوا خیلی ... |
| CNN | امروز یک روز خوب ... |
| SVM | امروز در شهر ... |

---

## Technical Details

### LSTM
- 2-layer stacked LSTM with dropout between layers
- LayerNorm before the output projection
- Hidden state is warmed up on the prompt tokens before generation

### GPT-style Transformer
- Decoder-only architecture using `nn.TransformerEncoder` with a causal mask
- Pre-LayerNorm (more stable than Post-LayerNorm)
- Weight tying between the input embedding and the output projection

### Dilated Causal CNN
- Stack of 4 dilated causal Conv1d blocks (dilation = 1, 2, 4, 8)
- Receptive field: `1 + (3-1) × (1+2+4+8) = 31 tokens` ≈ full sequence length
- Left-only padding ensures no future token leakage

### SVM
- Context window of 5 token IDs converted to a string feature
- TF-IDF with character n-grams (2–4) over token-ID strings
- LinearSVC wrapped in `CalibratedClassifierCV` to enable probability-based sampling
- Restricted to the top-3000 most frequent tokens as prediction targets