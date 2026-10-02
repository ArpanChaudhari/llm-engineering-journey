# Day 13: Tokenizers & Chat Templates

## Overview

LLMs do not directly understand human text. They work with numbers.

A **Tokenizer** converts text into token IDs that the model can process and converts the model's token IDs back into text.

```text
Text → Tokenizer → Token IDs → LLM → Token IDs → Tokenizer → Text
```

---

## 🎯 Learning Objectives

- Understand what a tokenizer does.
- Understand sub-word tokenization.
- Use `encode()`, `decode()`, and `batch_decode()`.
- Understand why each model has its own tokenizer.
- Understand `apply_chat_template()`.
- See how StarCoder2 tokenizes programming code.

---

# 1. What is a Tokenizer?

### Definition

> A tokenizer converts text into smaller pieces called **tokens** and maps those tokens to integer IDs.

Modern LLMs commonly use **sub-word tokenization**.

For example:

```text
Tokenization
     ↓
Token + iza + tion
```

A common approach is **BPE (Byte Pair Encoding)**.

### Why sub-word tokens?

- Whole-word tokenization requires a very large vocabulary.
- Character-level tokenization creates very long sequences.
- Sub-word tokenization provides a balance between vocabulary size and flexibility.

---

# 2. Encoding & Decoding

## Encoding

**Text → Token IDs**

```python
tokens = tokenizer.encode(text)
```

Example:

```text
"I am excited"
       ↓
[40, 1097, 12304]
```

## Decoding

**Token IDs → Text**

```python
text = tokenizer.decode(tokens)
```

Example:

```text
[40, 1097, 12304]
       ↓
"I am excited"
```

## Batch Decode

`batch_decode()` can be used to inspect multiple token pieces and see how the tokenizer split the original text.

```text
Text
 ↓
Tokens
 ↓
Individual decoded pieces
```

---

# 3. Word ≠ Token

A word does not always correspond to one token.

For example:

```text
Tokenizers
    ↓
" Token" + "izers"
```

And:

```text
LLM
 ↓
" L" + "LM"
```

A token can contain:

- A complete word
- Part of a word
- A space + word/character
- Special symbols

---

# 4. Vocabulary

### Definition

> A vocabulary is the collection of tokens known by a tokenizer.

Each token has an integer ID:

```text
Token ID → Token

40    → "I"
1097  → " am"
12304 → " excited"
```

The **vocabulary size** is the total number of tokens in that vocabulary.

---

# 5. Why Does Every Model Need Its Own Tokenizer?

Different models are trained with different vocabularies and token-ID mappings.

For example:

```text
Llama Token ID 100 → Token A
Qwen Token ID 100  → Token B
```

Therefore:

> **Token IDs are tokenizer/model specific.**

You should always use the tokenizer associated with the model.

```python
tokenizer = AutoTokenizer.from_pretrained(
    "model-name"
)
```

Do not encode text with one model's tokenizer and pass those IDs directly to another model.

---

# 6. Loading a Tokenizer

Hugging Face provides `AutoTokenizer`:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "model-name"
)
```

This loads the appropriate tokenizer configuration for the selected model.

---

# 7. Chat Templates

Chat/Instruct models are trained using a specific conversation format.

A conversation can be represented simply as:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Tell me a joke."
    }
]
```

Different models may require different internal formats.

For example, one model may use special tokens such as:

```text
<|system|>
<|user|>
<|assistant|>
```

while another model may use a different format.

---

# 8. `apply_chat_template()`

Hugging Face provides:

```python
tokenizer.apply_chat_template()
```

It automatically converts the message list into the format expected by that specific chat model.

Example:

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

### Important parameters

**`tokenize=False`**

Returns the formatted prompt as text instead of token IDs.

**`add_generation_prompt=True`**

Adds the appropriate prompt ending so the model knows it should generate the assistant's response.

### Main idea

```text
Messages
   ↓
apply_chat_template()
   ↓
Model-specific prompt format
   ↓
LLM
```

---

# 9. Different Models, Different Tokenization

The same text can be tokenized differently by different models.

Examples from this study:

- Llama 3.1
- Phi-3
- Qwen2
- StarCoder2

For example:

```text
Same sentence
      ↓
 ┌────┼────┐
Llama Phi Qwen
 ↓     ↓    ↓
Different tokens / IDs
```

This is why tokenizer compatibility matters.

---

# 10. StarCoder2 — Code Tokenization

StarCoder2 is a coding-focused model.

Its tokenizer is designed to handle programming text, including:

- Indentation
- Spaces
- Newlines
- Symbols
- Code syntax

Example:

```python
def hello_world(person):
    print("Hello", person)
```

You can inspect its tokens using:

```python
tokens = tokenizer.encode(code)

for token in tokens:
    print(repr(tokenizer.decode([token])))
```

Using `repr()` makes spaces and newline characters easier to see.

---

# 11. Key Concepts

| Concept | Meaning |
|---|---|
| Tokenizer | Converts text ↔ tokens/IDs |
| Token | Small piece of text |
| Token ID | Integer representing a token |
| Encoding | Text → Token IDs |
| Decoding | Token IDs → Text |
| Vocabulary | Collection of tokens |
| Vocabulary Size | Number of tokens |
| Chat Template | Model-specific conversation format |
| `encode()` | Text → IDs |
| `decode()` | IDs → Text |
| `batch_decode()` | Decode multiple token sequences |
| `apply_chat_template()` | Format chat messages |

---

Remember:

1. **Word ≠ Token**
2. One word can contain multiple tokens.
3. A token can include a leading space.
4. Token IDs are specific to a tokenizer.
5. Always use the tokenizer associated with the model.
6. `encode()` converts text to IDs.
7. `decode()` converts IDs back to text.
8. Chat models use model-specific chat templates.
9. `apply_chat_template()` handles the required chat formatting.
10. Code models such as StarCoder2 also tokenize spaces, indentation, and syntax.

---

# 13. Interview Questions

### Q1. What is a tokenizer?

**Answer:**  
A tokenizer converts text into tokens and maps those tokens to integer IDs that an LLM can process.

### Q2. Why do LLMs use sub-word tokenization?

**Answer:**  
It provides a balance between vocabulary size and sequence length while allowing the model to handle uncommon or new words through smaller pieces.

### Q3. Can two different models use the same token IDs?

**Answer:**  
Token IDs are tokenizer-specific. The same ID can represent different tokens in different models.

### Q4. What is `apply_chat_template()`?

**Answer:**  
It converts structured chat messages into the exact prompt format expected by a specific instruction/chat model.

### Q5. Why shouldn't we use one model's tokenizer with another model?

**Answer:**  
Each model can have a different vocabulary and token-ID mapping, so the second model may interpret the IDs incorrectly.