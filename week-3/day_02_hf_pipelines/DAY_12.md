# Day 2: Hugging Face APIs & Pipelines

## 1. Two API Levels of Hugging Face

### High-Level API

**Definition:**

> A high-level API provides a simple interface that hides much of the underlying model-processing complexity.

The main example is:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love Hugging Face!")
```

**Think:** High-Level API = less code and easier model usage.

### Low-Level API

**Definition:**

> A low-level API gives more direct control over components such as the tokenizer, model, inputs, and outputs.

Example:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("model_name")
model = AutoModel.from_pretrained("model_name")

inputs = tokenizer("Hello", return_tensors="pt")
outputs = model(**inputs)
```

**Think:** Low-Level API = more control and more code.

| High-Level | Low-Level |
|---|---|
| `pipeline()` | `AutoTokenizer` + `AutoModel` |
| Less code | More code |
| Easier | More control |
| Quick prototyping | Custom workflows |

---

# 2. High-Level API — `pipeline()`

### Definition

> `pipeline()` is a Hugging Face Transformers high-level API that provides an easy way to use pre-trained models for different AI tasks.

Basic structure:

```python
pipeline(task)
```

or:

```python
pipeline(
    task,
    model="model_name",
    device="cuda"
)
```

The pipeline handles much of:

```text
Input → Pre-processing → Model → Post-processing → Output
```

---

# 3. Setup

The notebook uses:

```python
!pip install -qU transformers datasets diffusers
```

Important imports:

```python
import torch
from huggingface_hub import login
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from IPython.display import Audio
```

Hugging Face authentication uses the stored `HF_TOKEN`.

```python
hf_token = userdata.get("HF_TOKEN")
login(hf_token)
```

---

# 4. Sentiment Analysis

### Definition

Sentiment Analysis classifies text based on sentiment, such as positive or negative.

```python
classifier = pipeline(
    "sentiment-analysis",
    device="cuda"
)

result = classifier(
    "I love using HuggingFace's Transformers library!"
)
```

### Custom Model

```python
classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device="cuda"
)
```

**Key idea:** `pipeline()` can use a default model or a model specified with `model=`.

---

# 5. Named Entity Recognition (NER)

### Definition

NER identifies entities such as:

- Person (`PER`)
- Location (`LOC`)
- Organization (`ORG`)

```python
ner = pipeline(
    "ner",
    aggregation_strategy="simple",
    device="cuda"
)
```

Example:

```python
result = ner(
    "Barack Obama was the 44th president of the United States."
)
```

`aggregation_strategy="simple"` combines related tokens into a single entity.

---

# 6. Text Generation

### Definition

Text Generation generates new text by continuing a given prompt.

```python
generator = pipeline(
    "text-generation",
    device="cuda"
)

result = generator(
    "The future of artificial intelligence will",
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)
```

- `max_new_tokens` → maximum new tokens
- `do_sample=True` → enables sampling
- `temperature` → controls randomness

---

# 7. Fill-Mask

### Definition

Fill-Mask predicts a missing token using surrounding context.

```python
unmasker = pipeline(
    "fill-mask",
    device="cuda"
)

mask = unmasker.tokenizer.mask_token

result = unmasker(
    f"Hugging Face is a great {mask} for machine learning."
)
```

The mask token depends on the model:

```text
BERT → [MASK]
RoBERTa → <mask>
```

Using `unmasker.tokenizer.mask_token` gets the correct token automatically.

---

# 8. Zero-Shot Classification

### Definition

Zero-Shot Classification allows us to provide candidate labels at runtime without creating a specific classifier for those labels.

```python
classifier = pipeline(
    "zero-shot-classification",
    device="cuda"
)

result = classifier(
    "The new iPhone has an incredible camera.",
    candidate_labels=[
        "technology",
        "sports",
        "politics",
        "entertainment"
    ]
)
```

**Key idea:**

```text
Text + Candidate Labels
          ↓
      Model scores
          ↓
   Relevant labels
```

---

# 9. Text-to-Speech

### Definition

Text-to-Speech (TTS) converts written text into spoken audio.

```python
tts = pipeline(
    "text-to-speech",
    model="facebook/mms-tts-eng",
    device="cuda"
)
```

Generate speech:

```python
speech = tts(
    "Hello, this is a test of the HuggingFace text-to-speech pipeline."
)
```

Play it:

```python
Audio(
    speech["audio"],
    rate=speech["sampling_rate"]
)
```

---

# 10. `device="cuda"`

Several pipelines use:

```python
device="cuda"
```

This runs the model on a CUDA-compatible GPU.

---

# 11. Transformers 4.x → 5.x Changes

| Old | Modern |
|---|---|
| `grouped_entities=True` | `aggregation_strategy="simple"` |
| `max_length` | `max_new_tokens` |

`max_new_tokens` specifies how many **new tokens** the model should generate after the prompt.

---

# 12. Key Things to Remember

```text
Hugging Face
     ↓
Two API Levels
     ↓
High-Level → pipeline()
Low-Level  → tokenizer + model
```

### Main pipeline tasks

| Task | Purpose |
|---|---|
| `sentiment-analysis` | Sentiment |
| `ner` | Named entities |
| `text-generation` | Generate text |
| `fill-mask` | Predict missing token |
| `zero-shot-classification` | Custom-label classification |
| `text-to-speech` | Generate speech |

### Most important concepts

1. `pipeline()` is a **High-Level API**.
2. `AutoTokenizer` + `AutoModel` provide a more **Low-Level approach**.
3. `model=` lets us choose a specific model.
4. `device="cuda"` runs inference on GPU.
5. `max_new_tokens` controls generated output length.
6. `aggregation_strategy="simple"` combines NER tokens.
7. The tokenizer provides the correct Fill-Mask token.

---

# 13. Interview Questions

### Q1. What is Hugging Face `pipeline()`?

**Answer:**  
`pipeline()` is a high-level API that simplifies the use of pre-trained models by handling much of the preprocessing, model execution, and post-processing.

### Q2. What are the two API levels?

**Answer:**  
High-Level APIs such as `pipeline()` provide simplicity, while Low-Level APIs such as `AutoTokenizer` and `AutoModel` provide more direct control.

### Q3. What is Zero-Shot Classification?

**Answer:**  
It classifies text using candidate labels provided at runtime without requiring a task-specific classifier for those exact labels.

### Q4. Why use `device="cuda"`?

**Answer:**  
It tells the pipeline to use a CUDA-compatible GPU for model execution.