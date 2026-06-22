# Next Word Prediction using LSTM

## Overview

This project implements a Next Word Prediction model using Natural Language Processing (NLP) and Deep Learning. The model is trained on a dataset of quotes and learns the contextual relationships between words to predict the most likely next word given an input sequence.

The pipeline includes text preprocessing, tokenization, sequence generation, padding, one-hot encoding, and training an LSTM-based neural network using TensorFlow/Keras.

---
 ## Screenshots
<img width="1027" height="567" alt="image" src="https://github.com/user-attachments/assets/307c9be4-e52a-448e-81d4-a1d12245950d" />

---

## Dataset

The model is trained on a collection of quotes stored in `qoute_dataset.csv`.

### Data Preparation

The following preprocessing steps are performed:

1. Load quotes from the dataset.
2. Convert all text to lowercase.
3. Remove punctuation characters.
4. Tokenize text using Keras Tokenizer.
5. Build a vocabulary of the top 10,000 words.

Example:

**Original Text**

```text
Success is not final, failure is not fatal.
```

**After Preprocessing**

```text
success is not final failure is not fatal
```

---

## Tokenization

The tokenizer converts words into integer indices.

Example:

```text
success is not final failure
```

may become

```python
[152, 4, 31, 827, 91]
```

A vocabulary size of 10,000 words is used.

```python
vocab_size = 10000
```

---

## Sequence Generation

Training samples are generated using a sliding-window approach.

For a sentence:

```text
success is not final failure is not fatal
```

Generated training examples:

| Input Sequence               | Target Word |
| ---------------------------- | ----------- |
| success                      | is          |
| success is                   | not         |
| success is not               | final       |
| success is not final         | failure     |
| success is not final failure | is          |

This allows the model to learn the probability distribution of the next word given previous words.

---

## Padding

Since input sequences have varying lengths, pre-padding is applied to ensure uniform sequence length.

Example:

```python
[152, 4]
```

becomes

```python
[0, 0, 0, 152, 4]
```

All sequences are padded to the maximum sequence length found in the dataset.

```python
padding='pre'
```

---

## Label Encoding

Target words are converted into one-hot encoded vectors.

Example:

```python
Target Word Index = 345
```

becomes

```python
[0, 0, 0, ..., 1, ..., 0]
```

where the vector length equals the vocabulary size (10,000).

---

## Model Architecture

The model consists of three layers:

### 1. Embedding Layer

Transforms integer word indices into dense vector representations.

```python
Embedding(
    input_dim=10000,
    output_dim=50
)
```

### 2. LSTM Layer

Captures sequential dependencies and contextual information.

```python
LSTM(
    units=128
)
```

### 3. Output Layer

Predicts the probability distribution over the entire vocabulary.

```python
Dense(
    units=10000,
    activation='softmax'
)
```

### Architecture Summary

```text
Input Sequence
       │
       ▼
Embedding Layer (50 Dimensions)
       │
       ▼
LSTM Layer (128 Units)
       │
       ▼
Dense Layer (10000 Units, Softmax)
       │
       ▼
Predicted Next Word
```

---

## Model Compilation

The model is compiled using:

```python
optimizer='adam'
loss='categorical_crossentropy'
metrics=['accuracy']
```

### Components

* **Adam Optimizer** for adaptive gradient optimization.
* **Categorical Crossentropy Loss** for multi-class classification.
* **Accuracy** as the evaluation metric.

---

## Training

The model is trained on the generated input-target pairs.

```python
lstm_model.fit(
    X_padded,
    y_one_hot,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)
```

During training, the model learns contextual patterns and word relationships from the quote dataset.

---

## Word Prediction

After training:

1. Create a reverse mapping from index to word.
2. Tokenize the input text.
3. Pad the sequence.
4. Pass it through the model.
5. Select the word with the highest probability.

Example:

### Input

```text
success is not
```

### Predicted Output

```text
final
```

### Generated Sentence

```text
success is not final
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* TensorFlow
* Keras
* LSTM Networks
* Natural Language Processing (NLP)

---

## Project Workflow

```text
Dataset
   │
   ▼
Text Cleaning
   │
   ▼
Tokenization
   │
   ▼
Sequence Generation
   │
   ▼
Padding
   │
   ▼
One-Hot Encoding
   │
   ▼
Embedding Layer
   │
   ▼
LSTM Layer
   │
   ▼
Softmax Output Layer
   │
   ▼
Next Word Prediction
```

---

## Learning Outcomes

* Text preprocessing and cleaning
* Tokenization and vocabulary creation
* Sequence generation for language modeling
* Sequence padding techniques
* One-hot encoding for classification tasks
* Building and training LSTM networks
* Next-word prediction using deep learning
* End-to-end NLP pipeline development
