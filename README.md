# 🤖 LLM Engineering Journey — Local & Free

Welcome to my portfolio repository documenting my learning journey through Ed Donner's Udemy course: **"LLM Engineering: Master AI and Large Language Models"**.

This repository showcases my solutions, rewritten from scratch to be **fully local, cost-free, and beginner-friendly**.

---

## 💡 About This Repository

Instead of using paid cloud APIs (like OpenAI or Anthropic), **all code in this repository runs entirely on my local machine** using **Ollama**.

* **No API Keys Required:** Free to run, with zero cloud hosting bills.
* **Privacy-First:** All prompts and generation occur completely offline.
* **Lightweight Models:** Uses Google's `gemma3:270m` and Meta's `llama3.2:1b`.
* **Beginner-Friendly Code:** Focuses on clean, readable Python and straightforward notebooks (no over-engineered helper packages).

---

## 🛠️ Technology Stack

* **Language:** Python 3.11+
* **Execution:** Jupyter Notebooks (`.ipynb`)
* **Local Inference Engine:** [Ollama](https://ollama.com)
* **LLMs Used:**
  * `llama3.2:1b` (1.2 Billion parameters — Meta's ultra-fast assistant)
  * `gemma3:270m` (270 Million parameters — Google's lightweight model)
* **Libraries:**
  * `ollama` (Official Python library for local model inference)
  * `tiktoken` (OpenAI's local tokenization engine for counting tokens)

---

## 📁 Repository Structure

```
LLM/
├── 📓 day_01_ollama_basics/
│   └── day_01_ollama_basics.ipynb       # Hello World, Roles, Memory, Benchmarks
│
├── 📓 day_02_models_and_prompts/
│   └── day_02_models_and_prompts.ipynb  # Temperature controls, JSON, Templates
│
├── 📓 day_03_prompt_engineering/
│   └── day_03_prompt_engineering.ipynb  # Few-shot, Chain-of-Thought, AI Code Reviewer
│
├── 📓 day_04_tokens_and_context/
│   └── day_04_tokens_and_context.ipynb  # Tokenization, tiktoken, strawberry edge cases
│
├── requirements.txt                     # All project dependencies
├── ROADMAP.md                           # Learning roadmap and progress tracker
└── README.md                            # This file
```

---

## 📓 What Each Day Builds

### 🦙 Day 1: Hello, LLMs! — First Contact with Ollama
Introduces the core concepts of local LLMs.
* **Code Built:** Standard Ollama API chat calls, system/user/assistant message arrays, a step-by-step chat history manager to give the AI memory, and a direct latency comparison between `llama3.2:1b` and `gemma3:270m`.

### 🧠 Day 2: LLM API Parameters & Prompt Designs
Controls the style and predictability of model output.
* **Code Built:** Temperature variance analysis (0.0 vs 1.0), extracting formatted database facts into a Python dictionary using Ollama's native JSON output mode, and building translation utilities using reusable prompt templates.

### ✍️ Day 3: Prompt Engineering Mastery
Masters advanced prompt instructions.
* **Code Built:** Few-shot comment classification using emoji labels, Chain-of-Thought (CoT) triggers to solve logic and math word problems, and an automated system-prompt-driven **AI Code Reviewer** that critiques bad code snippets.

### 🔢 Day 4: Tokens, Context Windows, and LLM Internals
Explores how models see and read text.
* **Code Built:** Word-to-token encoding and decoding using `tiktoken`, context window size budgeting, and a demonstration testing the limits of LLM spelling logic (the "strawberry" letter-counting challenge).

---

## ⚡ Quick Start (Run locally)

### 1. Prerequisites
Install **Ollama** on your computer from [ollama.com](https://ollama.com). Open your terminal/command prompt and download the models:
```bash
ollama pull llama3.2:1b
ollama pull gemma3:270m
```

### 2. Install Python Dependencies
Create a virtual environment and install the required libraries:
```bash
# Install packages
pip install -r requirements.txt
```

### 3. Run the Notebooks
Start Jupyter:
```bash
jupyter notebook
```
Navigate to any folder (e.g., `day_01_ollama_basics`) and open the `.ipynb` file to run the cells step-by-step!

*(Note: Make sure the Ollama app is running in the background of your system!)*

---

## 👤 Author

* **Arpan Chaudhari**
* *Actively learning AI/ML Engineering & Local LLM Applications*
