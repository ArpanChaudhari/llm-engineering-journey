# Day 4: Tokens, Context Windows & LLM Internals

## 📋 Overview
Day 4 pulls back the curtain on how Large Language Models actually process text. We learn about **Tokens** (the fundamental word pieces that models read) and how to count them locally using the **`tiktoken`** library. We cover **Context Windows** (the model's short-term memory limit), how developers budget tokens to prevent memory crashes, and explore the classic tokenizer **Edge Cases** (such as why LLMs struggle to count the letters in the word "strawberry").

---

## 🎯 Learning Objectives
* Understand what a token is and how character-to-token ratios work.
* Encode and decode text into integer token IDs using `tiktoken`.
* Learn how the Context Window behaves and how to build safety truncation checks in Python.
* Analyze why sub-word tokenization causes models to fail at character-level tasks.

---

## 📖 Key Concepts Explained (Beginner-Friendly)

### 1. What is a Token?
LLMs do not see text character-by-character or word-by-word. They use a **Tokenizer** to break text down into small, recurring chunks called **tokens**. 
* A token can be a whole word (e.g., `"robot"`), a word fragment (e.g., `"ro"`, `"bot"`), or a punctuation mark.
* Rule of thumb: **1 token ≈ 4 characters, or 0.75 words**.
* The tokenizer converts each token into a unique integer ID from its vocabulary library. The neural network only processes these list of numbers!

### 2. Context Window Limits
A model's **Context Window** is the maximum number of tokens it can read and write in a single prompt-response cycle. 
* E.g., `llama3.2:1b` has a context window of 128,000 tokens.
* E.g., `gemma3:270m` has a context window of 32,000 tokens.
* The context window is shared between the **input prompt** (system instructions, user text, few-shot examples, chat history) and the **output response** (what the AI generates). If the total sum exceeds this limit, the API calls will fail or the model will forget the beginning.

### 3. Tokenizer Edge Cases: The "Strawberry" Problem
LLMs famously fail at questions like *"How many 'r's are in the word 'strawberry'?"* (often replying with "two").
This occurs because of **sub-word tokenization**. The model's tokenizer splits `"strawberry"` into two token IDs representing `"straw"` and `"berry"`. The model's brains only process these two tokens—it never actually sees the letters `s-t-r-a-w-b-e-r-r-y`! To count letters, the model has to guess from training patterns, which often leads to errors.

---

## 💻 Code Walkthrough (Simplified)

### 1. Tokenizing and Decoding with tiktoken
```python
import tiktoken

# Load the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

text = "Hello world! LLMs are cool."

# Convert text to token IDs (integers)
token_ids = tokenizer.encode(text)
print(token_ids) # Output: [9906, 1917, 0, 15150, 482, 3591, 13]

# Show the text chunk that maps to each token ID
for token_id in token_ids:
    print(f"ID {token_id} -> '{tokenizer.decode([token_id])}'")
```

### 2. Context Window Safety Check
Before calling an LLM, developers verify that the prompt length fits within the safety limits to avoid server crashes.
```python
def check_safety_limit(prompt, safety_limit=2000):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_count = len(tokenizer.encode(prompt))
    
    if token_count <= safety_limit:
        return True, f"✅ Fits: {token_count} tokens."
    else:
        return False, f"❌ Too long: {token_count} tokens. Safety limit is {safety_limit}."

prompt_text = "This is some input text. " * 300
is_safe, message = check_safety_limit(prompt_text)
print(message)
```

---

## ❓ Interview Questions & Answers

#### Q1: Why do LLMs use sub-word tokenization instead of character-level or word-level tokenization?
**Answer:** 
* *Character-level* makes token list sequences extremely long, which exhausts context windows quickly.
* *Word-level* would require a vocabulary of millions of words to avoid "unknown word" errors, which makes the model's vocabulary layers too massive.
* *Sub-word tokenization* (like BPE or WordPiece) is the perfect middle-ground: it represents common words as single tokens, but breaks rare words down into small syllables (e.g. `"antidisestablishmentarianism"`), allowing a compact vocabulary (like 100,000 tokens) to represent any text.

#### Q2: How does the cost structure of paid APIs (like OpenAI) relate to tokenization?
**Answer:** Commercial LLM APIs bill you directly per 1 million input (prompt) and output (completion) tokens. Output tokens are almost always more expensive because they require sequential generation steps. Understanding tokenization helps engineers write concise prompts to optimize running costs.

#### Q3: How do you solve the "strawberry" spelling error in production?
**Answer:** Since LLMs cannot see characters directly, you can:
1. **Instruct the model to think step-by-step and spell out the word:** E.g., *"Spell 'strawberry' hyphenated: s-t-r-a-w-b-e-r-r-y. Now count the 'r's."* This forces it to generate character tokens.
2. **Use external Python code:** Write a simple Python function (`word.count('r')`) and let the LLM trigger it using **tool use / function calling** (which we learn in Week 2).

---

## 📝 Resume Bullet Points
* *Visualized and analyzed tokenization processes using tiktoken to audit input/output character-to-token ratios, optimizing API call payloads.*
* *Designed and implemented context window safety validators in Python to monitor token budgets and prevent model context overflow crashes.*
* *Analyzed tokenizer edge cases and sub-word segmentation limitations, implementing prompt mitigation strategies to resolve character-level logical errors.*
