# SkimLit📄🔥  
### Token + Character + Positional (Tribrid) Deep Learning Model

---

## 📄 Research Paper Reference

This project is inspired by and based on the research paper:

**PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts**  
Franck Dernoncourt, Ji Young Lee

### 🔗 Paper Link
- https://arxiv.org/abs/1710.06071
- https://github.com/Franck-Dernoncourt/pubmed-rct

---

## 📌 Project Overview

This project implements a **state-of-the-art neural network architecture** to classify sentences from **biomedical research abstracts** into their rhetorical roles using the **PubMed RCT (Randomized Controlled Trial) dataset**.

Each sentence in an abstract is classified into one of the following categories:

- **OBJECTIVE**
- **METHODS**
- **RESULTS**
- **CONCLUSIONS**
- **BACKGROUND** *(dataset-dependent)*

The model uses a **Tribrid Architecture** combining:

1. **Token-level semantic embeddings**
2. **Character-level morphological embeddings**
3. **Positional information**

This approach closely follows modern NLP research for structured document understanding.

---



## 📂 Dataset

**PubMed 20k RCT Dataset**

- ~20,000 biomedical abstracts
- Numbers replaced with `@` to prevent memorization
- Each sentence labeled with its rhetorical role



---

## 🏗️ Model Architecture (Model-5)

### High-level architecture

Text
├─ Token Embedding (Universal Sentence Encoder)
├─ Character Embedding (BiLSTM)
├─ Line Number Encoding
└─ Total Lines Encoding
↓
Concatenation
↓
Dense + Dropout
↓
Softmax Output


---

## 🧠 Architecture Components

---

### 1️⃣ Token-Level Embeddings (Semantic Meaning)

- Uses **Universal Sentence Encoder (USE)**
- Generates a **512-dimensional vector** per sentence
- Captures sentence-level semantics

Mathematically:

\[
\mathbf{e}_{token} = f_{\text{USE}}(s) \in \mathbb{R}^{512}
\]

Projected to lower dimension:

\[
\mathbf{h}_{token} = \text{ReLU}(W_t \mathbf{e}_{token} + b_t)
\]

---

### 2️⃣ Character-Level Embeddings (Morphology)

- Sentences split into characters
- Characters embedded and passed through **Bidirectional LSTM**
- Handles:
  - Medical terms
  - Misspellings
  - Rare words

\[
\mathbf{h}_{char} = \text{BiLSTM}(\text{Embed}(c_1, \dots, c_n))
\]

Output dimension:
\[
\mathbf{h}_{char} \in \mathbb{R}^{64}
\]

---

### 3️⃣ Positional Embeddings (Document Structure)

#### a) Line Number Encoding
- One-hot encoded (depth = 15)
- Indicates sentence position

\[
\mathbf{p}_{line} = \text{ReLU}(W_l \cdot \text{OneHot(line\_number)})
\]

#### b) Total Lines Encoding
- One-hot encoded (depth = 20)
- Indicates abstract length

\[
\mathbf{p}_{total} = \text{ReLU}(W_t \cdot \text{OneHot(total\_lines)})
\]

---

### 4️⃣ Tribrid Fusion

All representations are concatenated:

\[
\mathbf{h}_{combined} =
[\mathbf{h}_{token} \| \mathbf{h}_{char} \| \mathbf{p}_{line} \| \mathbf{p}_{total}]
\]

Followed by dense transformation and dropout:

\[
\mathbf{z} = \text{ReLU}(W_c \mathbf{h}_{combined} + b_c)
\]

---

### 5️⃣ Output Layer

Final classification using Softmax:

\[
\hat{y} = \text{Softmax}(W_o \mathbf{z} + b_o)
\]

---

## 🎯 Loss Function

The model uses **Categorical Cross-Entropy with Label Smoothing**:

\[
\mathcal{L} =
- \sum_{i=1}^{C} \tilde{y}_i \log(\hat{y}_i)
\]

Label smoothing:

\[
\tilde{y}_i =
(1 - \epsilon) y_i + \frac{\epsilon}{C}
\]

This improves generalization and reduces over-confidence.

---

## ⚙️ Training Strategy

- **Optimizer:** Adam
- **Regularization:**
  - Dropout (0.5)
  - Label smoothing (0.2)
- **Callbacks:**
  - EarlyStopping (patience = 3)
  - ModelCheckpoint (best weights only)
- **Training Setup:**
  - Partial-epoch training (10%) for faster experimentation

---

## 📊 Training Results

| Epoch | Validation Accuracy | Validation Loss |
|------|---------------------|----------------|
| 1 | 79.5% | 0.99 |
| 5 | 83.3% | 0.93 |
| 10 | 84.5% | 0.91 |
| 19 | **85.5%** | **0.90** |

### Observations
- Stable convergence
- No severe overfitting
- EarlyStopping triggered correctly

---

