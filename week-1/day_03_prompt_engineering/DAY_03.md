# Day 3: Prompt Engineering Mastery

## 📋 Overview
Day 3 focuses on the art and science of **Prompt Engineering**. We learn how to guide models using examples (Few-shot prompting) to achieve strict formatting and high classification accuracy. We also cover **Chain of Thought (CoT)** prompting, which forces models to write down their logical reasoning step-by-step, greatly improving accuracy on math and logical reasoning tasks. Finally, we apply these concepts to build a practical **AI Code Reviewer** utility.

---

## 🎯 Learning Objectives
* Understand the difference between Zero-shot and Few-shot prompting and when to use each.
* Implement Few-shot prompts to enforce strict output schemas (like single emojis) without API configurations.
* Apply Chain of Thought (CoT) prompting to solve complex math and logic word problems.
* Design a system-prompt-driven utility that acts as an expert code critic.

---

## 📖 Key Concepts Explained (Beginner-Friendly)

### 1. In-Context Learning: Zero-Shot vs. Few-Shot
* **Zero-Shot Prompting:** You ask the model to do something directly without giving it any examples (e.g. *"Classify this review as positive or negative."*). This works well for simple tasks but often fails at strict formatting (the model might add polite preambles or explanations).
* **Few-Shot Prompting:** You provide the model with a few complete examples of input-output pairs inside the prompt before giving it the actual task. By seeing the pattern, the model learns the exact style, length, and structure of the response we want. This is called **In-Context Learning**.

### 2. Chain of Thought (CoT) Reasoning
When humans solve a math word problem, we write down intermediate steps (e.g. *"First, calculate A, then add B, which equals C"*). If we are forced to shout out the final answer immediately, we are much more likely to make a mistake.

LLMs generate text word-by-word. If they are forced to output the final answer in the very first token, they are using very few computational steps to "think". By adding the instructions **"Think step-by-step before writing the final answer"**, the model writes out its intermediate reasoning. The words it generates act as a "scratchpad" that guides the model to the correct final conclusion.

### 3. Role / Persona Prompting
By instructing the model: *"You are an expert copywriter"* or *"You are a senior developer"*, you steer the model's internal attention mechanisms toward a subset of its training data related to that domain. This changes the vocabulary, style, and quality of the response.

---

## 💻 Code Walkthrough (Simplified)

### 1. Few-Shot Sentiment Classifier (Single Emoji Output)
```python
import ollama

few_shot_prompt = """
Classify the comment sentiment using a single emoji. Examples:
Comment: "I love this book!" -> 😍
Comment: "It was okay." -> 😐
Comment: "Awful service." -> 😡

Now classify this one:
Comment: "My package was completely crushed on arrival!" ->"""

response = ollama.chat(
    model='llama3.2:1b',
    messages=[{'role': 'user', 'content': few_shot_prompt}]
)
print(response.message.content.strip()) # Output: 😡
```

### 2. Chain of Thought (CoT) Prompting
```python
import ollama

problem = "A box has 3 red marbles and 4 blue marbles. If we double the red marbles and remove 2 blue ones, how many marbles are left?"

# Trigger CoT by asking the model to think step-by-step
cot_prompt = f"Solve this problem. Think step-by-step:\n{problem}"

response = ollama.chat(
    model='llama3.2:1b',
    messages=[{'role': 'user', 'content': cot_prompt}]
)
print(response.message.content)
```

### 3. AI Code Reviewer Utility
```python
import ollama

system_prompt = """
You are a senior Python code reviewer. For any code provided:
1. Give a quality score from 1-10.
2. List 2 main bugs or areas of improvement.
3. Show the refactored, clean Python version.
"""

bad_code = "def add_elements(lst): return sum([x for x in lst if x is not None])"

response = ollama.chat(
    model='llama3.2:1b',
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Review this code:\n{bad_code}"}
    ]
)
print(response.message.content)
```

---

## ❓ Interview Questions & Answers

#### Q1: What is "In-Context Learning" (Few-Shot), and does it update the model's weights?
**Answer:** No. Few-shot prompting does not alter the model's underlying neural network parameters (weights). The learning happens temporarily within the prompt context window during inference. Once the API call finishes, the model "forgets" the examples.

#### Q2: How does Chain of Thought (CoT) prompting improve an LLM's logical reasoning?
**Answer:** LLMs predict next words sequentially. If a model must output the final answer immediately, it has only one forward pass of computation to solve the problem. If it writes its reasoning step-by-step, the model can reference its *own* generated words as additional context, allocating more computational steps (tokens) to solve the problem.

#### Q3: What is the main drawback of Few-Shot and Chain of Thought prompting in production?
**Answer:** Both techniques increase **token usage**. Few-shot prompts require sending several examples with every API call, which increases input token cost. CoT prompts require the model to output long explanations, which increases output token cost and latency (time-to-respond).

#### Q4: How do you handle cases where a Few-Shot model still outputs conversational text instead of just the target label?
**Answer:** You can combine few-shot prompting with:
1. **System prompt constraints** (e.g. *"Answer with ONLY the label. Do not explain."*).
2. **Deterministic settings** (set temperature to 0.0 to prevent creative ramblings).
3. **Structured outputs** (force JSON format).

---

## 📝 Resume Bullet Points
* *Mastered advanced prompt engineering patterns including few-shot in-context learning, role prompting, and chain of thought (CoT) reasoning to improve LLM accuracy on structured classification and logical tasks.*
* *Designed and built a system-prompt-driven automated AI Code Reviewer that evaluates Python scripts, identifies unpythonic syntax, and outputs optimized code refactoring suggestions.*
* *Reduced model hallucinations and formatting errors by structuring system personas and prompt constraints, ensuring reliable output formats across lightweight local models.*
