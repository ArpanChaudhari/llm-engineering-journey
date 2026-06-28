# Day 9: Function Calling & Database Integration

## 📋 Overview
Day 9 covers one of the most powerful features of modern Large Language Models: **Function Calling (Tools)**. By default, an LLM only knows information from its training data. By giving it "Tools", we allow the AI to pause its text generation, ask us to run a local Python function, read the result, and then continue generating a response. We built an AI Customer Support Assistant (FlightAI) and connected it to a real SQLite database to fetch live ticket prices.

---

## 🎯 Learning Objectives
* Understand the concept and JSON schema of LLM Tool Calling.
* Implement parallel tool calling using a `for` loop to process multiple requests simultaneously.
* Implement sequential tool calling using a `while` loop to allow the LLM to chain tool executions.
* Connect a local LLM tool to a real SQLite database to fetch dynamic data.

---

## 📖 Key Concepts Explained (Beginner-Friendly)

### 1. What is a "Tool" in AI?
A tool is simply a standard Python function (e.g., `get_ticket_price()`). We provide the LLM with a JSON dictionary that describes the function's name, its purpose, and the arguments it expects. When the user asks a question, the LLM analyzes it, determines if it needs to use the tool, and returns a `tool_calls` request instead of standard text.

### 2. Parallel vs. Sequential Tool Calls
* **Parallel Tool Calls (`for` loop):** Sometimes a user asks two questions at once (*"Price for London and Paris?"*). The LLM will return an array of multiple tool calls in a single response. We use a `for` loop in a helper function (`handle_tool_calls`) to run all of them at once.
* **Sequential Tool Calls (`while` loop):** Sometimes the LLM needs to use a tool, read the result, and then decide to use *another* tool before giving a final answer. If we only use an `if` statement, the bot stops after one tool. By replacing the `if` statement with a `while response.choices[0].finish_reason == "tool_calls":` loop, the LLM can keep calling tools over and over until it has gathered all the facts it needs.

### 3. SQLite Database Integration
Hardcoding data in a Python dictionary is not scalable. Instead, we created a local `.db` file using Python's built-in `sqlite3` library. We updated our Python tool to run a SQL `SELECT` query using the arguments provided by the LLM, ensuring the AI always has access to the most up-to-date, real-world data.

---

## ❓ Interview Questions & Answers

#### Q1: How does an LLM know when to use a tool instead of just answering a question?
**Answer:** When making an API call to the LLM, we pass a `tools` parameter containing the JSON schema of available functions. The LLM's internal routing (trained specifically for function calling) evaluates the user's prompt against the tool descriptions. If a tool is highly relevant and required to answer the prompt accurately, the LLM outputs a `finish_reason` of `tool_calls` instead of generating a text response.

#### Q2: Why is a `while` loop necessary for advanced tool calling architectures?
**Answer:** A `while` loop enables sequential tool chaining. In complex agents, the output of one tool might be required as the input for the next tool. A `while` loop continuously checks if the LLM's `finish_reason` is `tool_calls`. It executes the tools, appends the results to the message history, and re-prompts the LLM. The loop only breaks when the LLM finally outputs a text response, ensuring complex, multi-step reasoning can occur without premature termination.

#### Q3: What is the purpose of the `tool_call_id` when appending results back to the message history?
**Answer:** The `tool_call_id` maps the result of the Python function back to the specific request made by the LLM. Because an LLM can request multiple tool calls in parallel (e.g., requesting the price for two different cities simultaneously), the ID ensures the LLM knows exactly which result belongs to which request.

---
