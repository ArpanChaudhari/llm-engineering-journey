# Day 1: Hello, LLMs! — First Contact with Ollama

## 📋 Overview
Day 1 introduces the core concepts of working with Large Language Models (LLMs) locally. Instead of relying on paid cloud APIs (like OpenAI), we use **Ollama** to run open-weight models directly on our machine. We cover the standard chat completion format, message roles, conversation statelessness, and basic performance benchmarks.

---

## 🎯 Learning Objectives
* Understand how to configure and run local LLMs with Ollama.
* Master the standard Chat Completions message list structure.
* Understand the distinct purposes of the `system`, `user`, and `assistant` roles.
* Learn how to maintain chat history manually to give LLMs "memory".
* Analyze latency and output trade-offs between different local model sizes.

---

## 📖 Key Concepts Explained (Beginner-Friendly)

### 1. Local Inference (Ollama) vs. Cloud APIs
Traditional AI applications make requests to cloud servers (like OpenAI's GPT-4 or Anthropic's Claude). While convenient, cloud APIs cost money per token, require an active internet connection, and present privacy risks for sensitive data. 

**Ollama** is an open-source tool that packages model weights (like Meta's Llama or Google's Gemma) and runs them on your local CPU/GPU. Running models locally is:
* **100% Free:** No token charges or subscription fees.
* **Completely Private:** Your data never leaves your computer.
* **Offline-Ready:** You can develop and test without internet access.

### 2. Message Roles: System, User, and Assistant
Chat completions APIs represent conversations as an ordered list of message dictionaries. Each message has a specific `role` that dictates how the model processes it:
* **`system`:** The global instruction set. It defines the AI's persona, rules of engagement, tone, and constraints (e.g., *"Speak like a pirate and keep answers under 20 words"*). It is placed at the very beginning of the history.
* **`user`:** The human speaking. This is the prompt or question you write.
* **`assistant`:** The model's response. When building chat memory, the assistant's replies must be captured and appended back to the conversation list.

### 3. LLM Statelessness & Memory Construction
LLMs are **stateless**; they do not remember previous requests. Each time you make an API call, the server boots up a clean slate. 

To simulate a "chat thread" where the model remembers your name or earlier questions, you must build a history list in Python (e.g., `messages = []`) and append both your prompts (`user` role) and the model's responses (`assistant` role) to it. On every new turn, you send the **entire** updated list.

### 4. Model Sizes and Parameters (1B vs. 270M)
Models are benchmarked by their parameter count (weights):
* **`llama3.2:1b` (1.2 Billion parameters):** A larger model with deeper reasoning capabilities. It generates more detailed responses but takes slightly longer to run on consumer hardware.
* **`gemma3:270m` (270 Million parameters):** A tiny, highly optimized model. It runs extremely fast and uses very little RAM, making it perfect for simple classification or speed-critical tasks.

---

## 💻 Code Walkthrough (Simplified)

### 1. Standard API Call
```python
import ollama

response = ollama.chat(
    model='llama3.2:1b',
    messages=[{'role': 'user', 'content': 'What is recursion?'}]
)
# Access the text response directly
print(response.message.content)
```

### 2. Chat Memory Construction
```python
messages = []

# Turn 1
messages.append({'role': 'user', 'content': 'Hi, my name is John.'})
response = ollama.chat(model='llama3.2:1b', messages=messages)
messages.append({'role': 'assistant', 'content': response.message.content})

# Turn 2 (The model knows our name because we send the history list!)
messages.append({'role': 'user', 'content': 'What is my name?'})
response2 = ollama.chat(model='llama3.2:1b', messages=messages)
print(response2.message.content) # Output: "Your name is John."
```

---

## ❓ Interview Questions & Answers
Here are typical questions recruiters ask during AI Engineering interviews:

#### Q1: Why are LLM APIs designed to be stateless?
**Answer:** Stateless designs make LLM servers highly scalable. The server doesn't need to store gigabytes of conversation history for millions of active users. It simply takes the input text list, runs the prediction math, returns the output, and instantly frees up RAM.

#### Q2: What is the difference between the `system` role and the `user` role?
**Answer:** The `system` message acts as a set of rules and identity constraints that the model prioritizes globally. The `user` message is the active task or question. The model evaluates the `user` prompt through the lens of the `system` rules.

#### Q3: If a local model runs too slowly on a machine, what parameters or choices can you adjust?
**Answer:** You can:
1. **Choose a smaller model size** (e.g., switching from `llama3.2:3b` to `llama3.2:1b` or `gemma3:270m`).
2. **Adjust parameters** like limiting `max_tokens` to stop the model from generating long-winded answers.
3. **Offload computation** to a GPU if available, or run quantised model weights (which use compressed 4-bit or 8-bit precision instead of 16-bit).

#### Q4: What is a token, and how does it affect cost and context window boundaries?
**Answer:** A token is the basic character chunk that an LLM processes (approx. 4 characters or 0.75 words). Every model has a "context window" limit (e.g., 2,048 tokens). Sending prompts that exceed this limit causes the model to lose memory or crash. For commercial APIs, billing is directly calculated per 1 million input/output tokens.

#### Q5: When would you use a low temperature (0.0) versus a high temperature (1.0)?
**Answer:** Use **temperature 0.0** (deterministic) for tasks requiring factual accuracy, structured code syntax, or logical reasoning (like code generation or math). Use **temperature 1.0** (random/creative) for open-ended brainstorming, creative writing, or interactive chatbot personas.

---

## 📝 Resume Bullet Points
Add this to your resume to showcase today's work to recruiters:
* *Implemented local LLM deployment pipelines using Ollama to run lightweight open-weight models (`llama3.2:1b`, `gemma3:270m`) offline, reducing development costs to zero.*
* *Engineered modular chat history arrays to manage multi-turn memory state and conversation flow across stateless completions APIs.*
