# 🧠 SkimLit 📄🔥  
### Token + Character + Positional (Tribrid) Deep Learning Model

---

## 📄 Research Paper Reference

This project is inspired by the research paper:

**PubMed 200k RCT: A Dataset for Sequential Sentence Classification in Medical Abstracts**  
Franck Dernoncourt, Ji Young Lee (IJCNLP 2017)

🔗 Links:
- https://arxiv.org/abs/1710.06071  
- https://github.com/Franck-Dernoncourt/pubmed-rct

This paper introduced the task of **sequential sentence classification** and showed that
sentence **position + content** is crucial in medical abstracts.

---

## 📌 Project Overview

This project implements a **deep learning model** to classify sentences from
**biomedical research abstracts** into their rhetorical roles using the
**PubMed RCT dataset**.

Each sentence is classified into one of the following categories:

- **OBJECTIVE**
- **METHODS**
- **RESULTS**
- **CONCLUSIONS**
- **BACKGROUND** (dataset dependent)

The model follows a **Tribrid Architecture** that combines:

1. Token-level semantic meaning  
2. Character-level word morphology  
3. Positional information within the abstract  

This mirrors the structure of real medical papers.

---

## 📂 Dataset

**PubMed 20k RCT Dataset**

- ~20,000 biomedical abstracts
- Numbers replaced with `@` to reduce overfitting
- Each sentence labeled with its section type


---

## 🏗️ Model Architecture (Model-5)

### High-Level Architecture


Sentence
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
- Produces a **512-dimensional embedding per sentence**
- Captures semantic meaning of the full sentence

Conceptually:
- Each sentence `s` is converted into a dense vector `e_token`
- This vector is passed through a Dense + ReLU layer to reduce dimensionality

This helps the model understand *what the sentence is about*.

---

### 2️⃣ Character-Level Embeddings (Morphology)

- Sentences are split into characters
- Characters are embedded and processed by a **Bidirectional LSTM**
- Helps with:
  - Medical terminology
  - Rare words
  - Misspellings

The BiLSTM captures both **prefix and suffix patterns**, producing a compact
representation of word structure.

---

### 3️⃣ Positional Embeddings (Document Structure)

Medical abstracts follow a fixed structure.

#### a) Line Number Encoding
- One-hot encoded vector (depth = 15)
- Represents the sentence’s position in the abstract
- Example: early lines → OBJECTIVE, later lines → RESULTS

#### b) Total Lines Encoding
- One-hot encoded vector (depth = 20)
- Represents the overall abstract length

These features help the model learn **document flow**.

---

### 4️⃣ Tribrid Fusion

All representations are concatenated:

- Token embedding
- Character embedding
- Line number embedding
- Total lines embedding

The combined vector is passed through:
- Dense layer (ReLU)
- Dropout (0.5) for regularization

This allows the model to jointly reason about
**content + form + position**.

---

### 5️⃣ Output Layer

- Final Dense layer with **Softmax**
- Outputs probabilities for each section label

The predicted label is the class with the highest probability.

---

## 🎯 Loss Function

The model uses **Categorical Cross-Entropy with Label Smoothing**.

Why label smoothing?
- Prevents the model from becoming over-confident
- Improves generalization

Instead of forcing probability = 1 for the true class,
a small amount is distributed across other classes.

---

## ⚙️ Training Strategy

- **Optimizer:** Adam  
- **Regularization:**
  - Dropout = 0.5
  - Label smoothing = 0.2  
- **Callbacks:**
  - EarlyStopping (patience = 3)
  - ModelCheckpoint (save best weights only)
- **Training Setup:**
  - Partial-epoch training (10%) for faster iteration

---

## 📊 Training Results

| Epoch | Validation Accuracy | Validation Loss |
|------|---------------------|----------------|
| 1 | 79.5% | 0.99 |
| 5 | 83.3% | 0.93 |
| 10 | 84.5% | 0.91 |
| 19 | **85.5%** | **0.90** |


