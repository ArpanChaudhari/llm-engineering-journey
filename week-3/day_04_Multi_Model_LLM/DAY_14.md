# Day 14: Hugging Face Models & Multi-Model LLM Experiment

## Overview

Day 14 goes one level deeper than `pipeline()`.

Instead of letting Hugging Face handle everything automatically, we directly work with:

- `AutoTokenizer`
- `AutoModelForCausalLM`
- `model.generate()`
- `BitsAndBytesConfig`
- `TextStreamer`

The main flow is:

```text
Text
  ↓
Tokenizer
  ↓
Token IDs
  ↓
LLM / Transformer
  ↓
Output Token IDs
  ↓
Tokenizer.decode()
  ↓
Text
```

The notebook experiments with five instruction-tuned models:

| Variable | Model | Company | Parameters |
|---|---|---|---:|
| `LLAMA` | Llama-3.2-1B-Instruct | Meta | 1B |
| `PHI` | Phi-4-mini-instruct | Microsoft | ~4B |
| `GEMMA` | Gemma-3-270m-it | Google | 270M |
| `QWEN` | Qwen3-4B-Instruct | Alibaba | 4B |
| `DEEPSEEK` | DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek AI | 1.5B |

All are instruction-tuned models designed to follow instructions and generate conversational responses.

---

## 🎯 Learning Objectives

- Understand the low-level Transformers model API.
- Load a tokenizer using `AutoTokenizer`.
- Load a causal language model using `AutoModelForCausalLM`.
- Understand model parameters and GPU memory.
- Understand 4-bit quantization.
- Use `apply_chat_template()` before generation.
- Generate text using `model.generate()`.
- Decode model output back into text.
- Understand `TextStreamer`.
- Reuse one generation function for multiple models.
- Compare different LLMs using the same prompt.

---

# 1. Low-Level Model API

Previously:

```python
pipeline(...)
```

handled many steps automatically.

Now we manually control the process:

```text
Tokenizer
    ↓
Chat Template
    ↓
Model
    ↓
Generate
    ↓
Decode
```

This gives us more control over the model and its memory usage.

---

# 2. Install Required Libraries

```python
!pip install -qU transformers accelerate bitsandbytes
```

### Libraries

| Library | Purpose |
|---|---|
| `transformers` | Hugging Face models and tokenizers |
| `accelerate` | Efficient CPU/GPU model loading |
| `bitsandbytes` | Model quantization |
| `torch` | PyTorch deep-learning framework |

---

# 3. Imports

```python
from huggingface_hub import login
from google.colab import userdata

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig,
)

import torch
import gc
```

### Important APIs

- `AutoTokenizer` → loads the model's tokenizer.
- `AutoModelForCausalLM` → loads a causal language model.
- `TextStreamer` → streams generated text.
- `BitsAndBytesConfig` → configures quantization.
- `torch` → PyTorch.
- `gc` → helps release memory.

---

# 4. Hugging Face Login

Some models require Hugging Face authentication/access.

```python
hf_token = userdata.get("HF_TOKEN")

login(
    hf_token,
    add_to_git_credential=True
)
```

The token is stored in Colab Secrets instead of writing it directly in the notebook.

---

# 5. GPU and Memory

The notebook uses a Google Colab T4 GPU with approximately 16 GB VRAM.

```python
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )
```

### Why GPU memory matters?

LLMs can contain billions of parameters.

More parameters → more memory required.

---

# 6. Model Size & Parameters

A model's size is commonly described by its number of parameters.

```text
8B = 8 billion parameters
```

Approximate weight memory for an 8B model:

| Precision | Bytes / Parameter | Approx. Weight Memory |
|---|---:|---:|
| FP32 | 4 | ~32 GB |
| FP16 / BF16 | 2 | ~16 GB |
| INT8 | 1 | ~8 GB |
| INT4 | 0.5 | ~4 GB |

Actual GPU usage is higher because inference also needs memory for activations, KV cache, temporary tensors, etc.

---

# 7. 4-Bit Quantization

### Definition

> Quantization reduces the number of bits used to represent model weights, reducing memory usage.

The notebook configures 4-bit loading:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
```

### Important parameters

- `load_in_4bit=True` → load weights using 4-bit quantization.
- `bnb_4bit_use_double_quant=True` → reduces quantization overhead.
- `bnb_4bit_compute_dtype=torch.bfloat16` → dtype used during computation.
- `bnb_4bit_quant_type="nf4"` → uses NF4 quantization.

### Main idea

```text
Higher precision
      ↓
More memory

4-bit quantization
      ↓
Less memory
      ↓
Larger models can fit on smaller GPUs
```

---

# 8. Define the Models

```python
LLAMA = "meta-llama/Llama-3.2-1B-Instruct"

PHI = "microsoft/Phi-4-mini-instruct"

GEMMA = "google/gemma-3-270m-it"

QWEN = "Qwen/Qwen3-4B-Instruct-2507"

DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
```

The same generation workflow can be reused for all five models.

---

# 9. Define the Prompt

```python
messages = [
    {
        "role": "user",
        "content": "Tell a joke for a room of Data Scientists"
    }
]
```

This uses the standard chat-message structure.

---

# 10. Load the Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(LLAMA)

tokenizer.pad_token = tokenizer.eos_token
```

### `AutoTokenizer`

Loads the tokenizer associated with the selected model.

Remember:

> **Always use the tokenizer that belongs to the model.**

---

# 11. Apply the Chat Template

```python
input = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

Different instruction models use different special conversation formats.

For example, Llama uses special tokens such as:

```text
<|start_header_id|>
<|end_header_id|>
<|eot_id|>
```

`apply_chat_template()` automatically creates the correct format.

### `tokenize=False`

Returns the formatted prompt as text.

### `add_generation_prompt=True`

Adds the appropriate assistant-ending prompt so the model knows it should generate a response.

---

# 12. Convert the Prompt to PyTorch Tensors

```python
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt"
).to("cuda")
```

### `return_tensors="pt"`

Returns a PyTorch tensor.

### `.to("cuda")`

Moves the input tensor to the GPU.

So:

```text
Messages
   ↓
Chat Template
   ↓
Token IDs
   ↓
PyTorch Tensor
   ↓
GPU
```

---

# 13. Load the Model

```python
model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map="auto",
    quantization_config=quantization_config
)
```

### `AutoModelForCausalLM`

Loads a causal language model.

Causal language modeling works by predicting the next token:

```text
The capital of France is
                    ↓
                  Paris
```

### `device_map="auto"`

Automatically decides where model components should be placed.

### `quantization_config`

Loads the model using the configured 4-bit quantization.

---

# 14. Check Model Memory

```python
memory = model.get_memory_footprint() / 1e6

print(f"Memory footprint: {memory:,.1f} MB")
```

This helps us understand how much memory the loaded model occupies.

In the notebook, the 1B Llama model with 4-bit quantization used around **1 GB** of memory.

---

# 15. Looking Inside the Transformer

Printing:

```python
model
```

shows the neural-network architecture.

Important components:

| Component | Purpose |
|---|---|
| `embed_tokens` | Converts token IDs into vectors |
| `layers` | Transformer decoder blocks |
| `self_attn` | Attention mechanism |
| `mlp` | Feed-forward network |
| `norm` | Normalization |
| `lm_head` | Produces scores over the vocabulary |

Conceptually:

```text
Token IDs
   ↓
Embedding
   ↓
Transformer Layers
   ↓
LM Head
   ↓
Next-token probabilities
```

The notebook's Llama model contains **16 decoder layers** and shows `Linear4bit` layers, confirming that 4-bit quantization is active.

---

# 16. Generate Text

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=80
)
```

`model.generate()` is the main text-generation method.

### `max_new_tokens=80`

The model can generate at most 80 new tokens.

The output is still token IDs:

```text
Token IDs
```

It is not readable text yet.

---

# 17. Decode the Output

```python
tokenizer.decode(outputs[0])
```

Converts:

```text
Token IDs
   ↓
Text
```

Important:

> `generate()` returns the original prompt + the newly generated tokens together.

So `decode(outputs[0])` can contain both the prompt and response.

---

# 18. Free GPU Memory

Before loading another model:

```python
del model, inputs, tokenizer, outputs

gc.collect()

torch.cuda.empty_cache()
```

### Why?

A GPU has limited VRAM.

If the previous model remains loaded:

```text
Old Model
   +
New Model
   ↓
GPU Out Of Memory ❌
```

These steps help release unused memory.

---

# 19. Reusable `generate()` Function

Instead of repeating the same code for every model, the notebook creates:

```python
def generate(model, messages, quant=True, max_new_tokens=80):

    tokenizer = AutoTokenizer.from_pretrained(model)

    tokenizer.pad_token = tokenizer.eos_token

    input = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")

    streamer = TextStreamer(tokenizer)

    if quant:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=quantization_config
        ).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model
        ).to("cuda")

    outputs = model.generate(
        **input,
        max_new_tokens=max_new_tokens
    )

    return tokenizer.decode(outputs[0])
```

Now the entire process is reusable.

---

# 20. Run Phi-4 Mini

```python
generate(PHI, messages)
```

The function automatically:

```text
Load tokenizer
      ↓
Apply chat template
      ↓
Load model
      ↓
Apply quantization
      ↓
Generate response
      ↓
Decode response
```

---

# 21. Run Gemma

```python
generate(
    GEMMA,
    messages,
    quant=False
)
```

Gemma-3-270M is a small model, so the notebook loads it without the 4-bit configuration.

The output also demonstrates that different models use different chat-template formats.

---

# 22. Run Qwen

```python
generate(QWEN, messages)
```

Qwen3-4B is larger, so the notebook uses the default:

```python
quant=True
```

This loads it with 4-bit quantization.

---

# 23. Run DeepSeek-R1

```python
generate(
    DEEPSEEK,
    messages,
    quant=False,
    max_new_tokens=500
)
```

DeepSeek-R1 is treated differently in this experiment because it is a **reasoning model**.

The notebook uses:

```python
max_new_tokens=500
```

because the reasoning model can produce a longer reasoning process before the final answer.

Its output in the notebook contains:

```text
<think>
...
</think>

Final answer
```

---

# 24. Comparing Multiple Models

The same prompt is sent to:

```text
Llama
Phi
Gemma
Qwen
DeepSeek
```

This demonstrates an important LLM-engineering idea:

> The inference workflow can stay almost the same while the underlying model changes.

Different models can produce different:

- Responses
- Styles
- Chat-template formats
- Tokenizations
- Memory requirements
- Reasoning behavior

---

# 25. Complete LLM Inference Flow

This is the most important diagram from Day 14:

```text
User Message
     ↓
messages
     ↓
AutoTokenizer
     ↓
apply_chat_template()
     ↓
Token IDs / Tensor
     ↓
GPU
     ↓
AutoModelForCausalLM
     ↓
model.generate()
     ↓
Output Token IDs
     ↓
tokenizer.decode()
     ↓
Final Text
```

With optimization:

```text
                 ┌── Quantization → Reduce memory
                 │
Input → Tokenizer → Model → Generate → Decode → Output
                    │
                    ├── GPU
                    └── TextStreamer
```

---

# 26. Key Concepts

| Concept | Meaning |
|---|---|
| `AutoTokenizer` | Loads the correct tokenizer |
| `AutoModelForCausalLM` | Loads a causal language model |
| `apply_chat_template()` | Formats chat messages for a model |
| `model.generate()` | Generates new tokens |
| `decode()` | Converts token IDs to text |
| `TextStreamer` | Streams generated text |
| Quantization | Reduces model memory usage |
| `BitsAndBytesConfig` | Configures quantization |
| `device_map="auto"` | Automatically maps model components |
| `cuda` | Runs tensors/model on GPU |
| `get_memory_footprint()` | Checks model memory usage |
| `gc.collect()` | Runs Python garbage collection |
| `torch.cuda.empty_cache()` | Releases unused CUDA cache |

---

# 27. Key Takeaways

1. `pipeline()` hides many steps; low-level APIs give more control.
2. `AutoTokenizer` converts text into model-compatible tokens.
3. `apply_chat_template()` creates the correct format for instruction/chat models.
4. `AutoModelForCausalLM` loads a causal language model.
5. `model.generate()` produces new token IDs.
6. `tokenizer.decode()` converts output IDs back into text.
7. Quantization reduces model memory requirements.
8. 4-bit quantization is useful when GPU VRAM is limited.
9. Different models can use different chat formats.
10. GPU memory must be managed when loading multiple models.
11. The same inference function can be reused across different LLMs.
12. Reasoning models may require more generation tokens.

---

# 28. Interview Questions

### Q1. What is `AutoModelForCausalLM`?

**Answer:**  
It is a Hugging Face Auto class used to load causal language models that generate text by predicting the next token.

### Q2. Why do we use `apply_chat_template()`?

**Answer:**  
It converts structured chat messages into the exact conversation format expected by a specific instruction/chat model.

### Q3. What is quantization?

**Answer:**  
Quantization reduces the numerical precision used to store model weights, which reduces memory usage and allows larger models to run on limited GPUs.

### Q4. What does `load_in_4bit=True` do?

**Answer:**  
It loads model weights using 4-bit quantization to significantly reduce memory requirements.

### Q5. What does `model.generate()` return?

**Answer:**  
It returns generated token IDs. These IDs can be converted into readable text using `tokenizer.decode()`.

### Q6. Why do we call `torch.cuda.empty_cache()`?

**Answer:**  
It releases unused cached CUDA memory so that GPU memory can be reused when loading another model.

### Q7. Why use `device_map="auto"`?

**Answer:**  
It allows the Transformers/Accelerate stack to automatically place model components on available hardware such as GPU and CPU.

### Q8. Why can the same prompt produce different outputs from different models?

**Answer:**  
Different models have different architectures, training data, tokenizers, weights, and instruction-tuning, so their generated responses can differ.
