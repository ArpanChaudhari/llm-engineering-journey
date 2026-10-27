# Day 18: Code Generator --- Python to High-Performance Java

## Overview

Day 18 explores how to use LLMs to translate Python code into Java and
compare the code produced by different models.

The notebook defines a Python benchmark that approximates pi, sends that
code to three configured model IDs, displays the generated Java
programs, and attempts to compile and time them.

The main workflow is:

``` text
Python Benchmark
      ↓
System Prompt + User Message
      ↓
Selected LLM API
      ↓
Generated Java Code
      ↓
Display Java Versions
      ↓
Save .java Files
      ↓
Compile with javac
      ↓
Run with java and Measure Time
```

## Learning Objectives

-   Load API keys from Google Colab Secrets.
-   Use an OpenAI-compatible client interface with different provider
    base URLs.
-   Understand model IDs and API clients.
-   Write a reusable prompt for code translation.
-   Understand the purpose of `messages_for()` and `port()`.
-   Ask multiple configured models to translate the same Python code.
-   Display model-generated Java code in a notebook.
-   Save Java source files, compile them, and measure execution time.
-   Recognize limitations in a simple benchmark and how to make
    comparisons fairer.

------------------------------------------------------------------------

# 1. Project Architecture

The notebook has four main parts:

1.  **Setup:** import libraries, load keys, and create API clients.
2.  **Prompt and benchmark:** define the instructions and Python program
    to translate.
3.  **Code generation:** send the same Python program to each model and
    display the responses.
4.  **Execution comparison:** save the Java files, compile them, and
    time their execution.

``` text
Setup API clients
      ↓
Define system prompt
      ↓
Prepare messages with Python code
      ↓
Call each selected model
      ↓
Collect generated Java code
      ↓
Compile and run each program
      ↓
Compare runtime and output
```

------------------------------------------------------------------------

# 2. Import Libraries

The notebook imports these libraries:

``` python
import os
import subprocess

from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI
```

  -----------------------------------------------------------------------
  Library / Function                  Purpose
  ----------------------------------- -----------------------------------
  `os`                                Operating-system utilities and
                                      environment variables

  `subprocess`                        Runs external commands such as
                                      `javac` and `java`

  `load_dotenv`                       Loads environment variables from a
                                      `.env` file

  `Markdown`                          Formats output as Markdown inside
                                      the notebook

  `display`                           Displays formatted content in the
                                      notebook

  `OpenAI`                            Provides an OpenAI-compatible
                                      client interface for API requests
  -----------------------------------------------------------------------

The notebook later imports `time` to measure execution duration and
`google.colab.userdata` to read secrets.

------------------------------------------------------------------------

# 3. Load and Check API Keys

The notebook retrieves two API keys from Colab Secrets:

``` python
from google.colab import userdata

GROQ_API_KEY = userdata.get("GROQ_API_KEY")
GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")

api_keys = {
    "Google": GOOGLE_API_KEY,
    "Grok": GROQ_API_KEY,
}

for provider, api_key in api_keys.items():
    status = "available" if api_key else "not set (optional unless using this provider)"
    print(f"{provider}: {status}")
```

### What is happening?

-   `userdata.get()` retrieves a secret saved in Google Colab.
-   The loop checks whether each key exists.
-   It prints only the status, not the secret value.

**Important:** The dictionary labels are just labels for display. The
key named `GROQ_API_KEY` is used for Groq, even though the notebook's
status label says `"Grok"`.

Never hard-code or share API keys in notebook output or public
repositories.

------------------------------------------------------------------------

# 4. Connect to Model Providers

The notebook creates two clients using the `OpenAI` class:

``` python
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
groq_url = "https://api.groq.com/openai/v1"

gemini = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url=gemini_url
)

groq = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=groq_url
)
```

### Why use `OpenAI` for both?

The `OpenAI` Python library can communicate with services that support
its compatible chat-completions API. The client interface stays similar,
while the API key and base URL identify the provider.

-   `gemini` uses the Google Gemini API's OpenAI-compatible endpoint.
-   `groq` uses Groq's OpenAI-compatible endpoint.

The variable names are Python objects representing configured clients;
they are not themselves the AI models.

### Base URL vs. model ID

-   **Base URL:** tells the client which API service to contact.
-   **API key:** authenticates the request.
-   **Model ID:** tells that service which model to run.

A model ID must be supported by the provider behind the client being
used.

------------------------------------------------------------------------

# 5. Choose the Models

The notebook sets these model IDs:

``` python
OPENAI_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.6-flash"
GROQ_MODEL = "qwen/qwen3.8-27b"
```

These are the exact strings in the notebook. Model availability and
naming can change, so verify the IDs against the provider's current
model list if a request fails.

**Important detail from the code:** The variable name does not decide
the provider. In Step 8, `OPENAI_MODEL` is passed to the `groq` client,
not to a separate OpenAI client. Therefore, that request is sent to Groq
using the model ID `"openai/gpt-oss-120b"`. The notebook does not create
a direct OpenAI API client.

------------------------------------------------------------------------

# 6. Create the Python-to-Java Prompt

## 6.1 System prompt

``` python
system_prompt = """
Your task is to convert Python code into high-performance Java code.
Respond only with valid Java code. Do not provide markdown formatting or explanations, except for occasional comments.
"""
```

The system prompt sets the model's role and output expectations:

-   Translate Python into Java.
-   Aim for high performance.
-   Return Java code, not a long explanation.
-   Avoid Markdown formatting.

A prompt guides the model, but it does not guarantee that the generated
code will compile or be equivalent to the original program.

## 6.2 Prepare messages

``` python
def messages_for(python_code):
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Convert this Python code to Java:\n{python_code}"
        },
    ]
```

The chat API accepts a list of messages.

  Role       Purpose
  ---------- ---------------------------------
  `system`   Instructions for the model
  `user`     The actual task and Python code

### Why use a function?

`messages_for(python_code)` builds the same message structure every
time, while allowing the Python code to change. This avoids repeating
the same list of dictionaries for each model.

For example:

``` python
messages_for("print(10)")
messages_for("print(20)")
```

Both calls use the same system prompt but include different Python code.

------------------------------------------------------------------------

# 7. Define the Python Benchmark

The notebook uses this program as the translation task:

``` python
pi_benchmark_code = """
def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations + 1):
        j = i * param1 - param2
        result -= 1 / j
        j = i * param1 + param2
        result += 1 / j
    return result

result = calculate(200_000_000, 4, 1) * 4
print(f"Result: {result:.12f}")
"""
```

### What does it do?

1.  Defines `calculate()` with three parameters.
2.  Repeats a calculation for `200,000,000` iterations.
3.  Updates a floating-point result using two denominators on each loop.
4.  Multiplies the final value by `4`.
5.  Prints the result with 12 digits after the decimal point.

The calculation approximates pi using a series. The large iteration
count makes it a computationally intensive benchmark.

### Why use the same benchmark for every model?

Giving each model the same input makes the generated implementations
easier to compare. The output still needs to be checked to ensure that
each version performs the same calculation and produces equivalent
results.

------------------------------------------------------------------------

# 8. Create the `port()` Function

``` python
def port(client, model, python_code):
    response = client.chat.completions.create(
        model=model,
        messages=messages_for(python_code)
    )

    java_code = response.choices[0].message.content or ""

    # Strip markdown block formatting if returned by the model
    java_code = java_code.replace("```java", "").replace("```", "").strip()

    return java_code
```

This function handles the complete translation request.

### Step-by-step

1.  **Inputs:** receives a client, a model ID, and Python source code.
2.  **Prepare messages:** calls `messages_for(python_code)`.
3.  **Call the API:** sends the messages to the selected model.
4.  **Extract output:** reads the first response choice's message
    content.
5.  **Clean the response:** removes common Java Markdown code fences and
    surrounding whitespace.
6.  **Return:** gives the resulting Java code back to the caller.

### Why separate `messages_for()` and `port()`?

-   `messages_for()` prepares the request content.
-   `port()` sends that content to the API and returns the response.

This separation makes the message format reusable and the API call
easier to maintain.

The code removes common code fences, but it does not validate or repair
the Java program.

------------------------------------------------------------------------

# 9. Translate the Benchmark with Each Model

``` python
print(f"Translating with {OPENAI_MODEL}...")
openai_java = port(groq, OPENAI_MODEL, pi_benchmark_code)

print(f"Translating with {GEMINI_MODEL}...")
gemini_java = port(gemini, GEMINI_MODEL, pi_benchmark_code)

print(f"Translating with {GROQ_MODEL}...")
groq_java = port(groq, GROQ_MODEL, pi_benchmark_code)
```

The same Python benchmark is sent three times.

  Variable        Client passed   Model ID variable
  --------------- --------------- -------------------
  `openai_java`   `groq`          `OPENAI_MODEL`
  `gemini_java`   `gemini`        `GEMINI_MODEL`
  `groq_java`     `groq`          `GROQ_MODEL`

Each variable stores the Java source returned by that request.

**Remember:** `openai_java` is only a variable name here. Since the code
passes `groq` as its client, it does not mean the request is sent
directly to OpenAI.

------------------------------------------------------------------------

# 10. Display the Generated Java Code

``` python
from IPython.display import Markdown, display

display(Markdown(f"### OpenAI\n```java\n{openai_java}\n```"))
display(Markdown(f"### Gemini\n```java\n{gemini_java}\n```"))
display(Markdown(f"### Groq\n```java\n{groq_java}\n```"))
```

`display(Markdown(...))` renders the source code with a heading and Java
syntax block inside the notebook.

This is useful for inspecting differences such as:

-   Whether each answer has a valid Java class and `main()` method.
-   How the loop and floating-point calculations are translated.
-   Whether the output formatting is equivalent.
-   Whether one implementation uses a different algorithm.

Displaying code does not prove that it compiles, runs correctly, or is
faster.

------------------------------------------------------------------------

# 11. Save the Java Files

The notebook writes each response to a separate file:

``` python
with open("MainOpenAI.java", "w") as f:
    f.write(openai_java.replace("class Main", "class MainOpenAI"))

with open("MainGemini.java", "w") as f:
    f.write(gemini_java.replace("class Main", "class MainGemini"))

with open("MainGroq.java", "w") as f:
    f.write(groq_java.replace("class Main", "class MainGroq"))
```

### What does `open(..., "w")` do?

It opens a file for writing. If the file does not exist, Python creates
it; if it exists, its contents are overwritten.

### Why replace the class name?

Java requires a public class name to match its filename. The notebook
attempts to change `class Main` to a model-specific name so that the
three programs can be saved separately.

**Limitation:** This is a simple text replacement. It only works if the
generated code uses the expected class declaration. It may fail if a
model uses a different class name or a different declaration style.

------------------------------------------------------------------------

# 12. Compile the Java Files

``` python
print("Compiling Java files...")

subprocess.run(["javac", "MainOpenAI.java"])
subprocess.run(["javac", "MainGemini.java"])
subprocess.run(["javac", "MainGroq.java"])
```

### What is `javac`?

`javac` is the Java compiler. It converts `.java` source files into
`.class` bytecode files that can be run by the Java Virtual Machine
(JVM).

The command:

``` bash
javac MainOpenAI.java
```

compiles that Java source file.

### What is `subprocess.run()`?

It lets Python execute an external command, such as the Java compiler.

**Important:** In the notebook, the compiler calls do not use
`check=True`, so a failed compilation may not stop the next cells. For
more reliable benchmarking, check each compiler's return code and stop
if compilation fails.

Java must be installed and both `javac` and `java` must be available in
the environment's PATH.

------------------------------------------------------------------------

# 13. Measure Execution Time

The notebook measures each program using this pattern:

``` python
import time
import subprocess

start = time.time()
subprocess.run(["java", "MainOpenAI"], capture_output=True)
openai_time = time.time() - start

print(f"OpenAI Model: {openai_time:.4f} seconds")
```

The same pattern is repeated for the Gemini and Groq-generated programs.

### How it works

1.  `time.time()` records the start time.
2.  `subprocess.run()` launches the Java program and waits for it to
    finish.
3.  `time.time() - start` calculates the elapsed time.
4.  `:.4f` formats the duration to four decimal places.

### What does `capture_output=True` do?

It captures the program's standard output and standard error rather than
displaying them directly. The notebook's timing code does not print the
captured result.

### Important benchmarking limitations

-   The notebook runs each Java program **once**, even though its
    Markdown says to execute them multiple times.
-   The measured duration includes JVM startup, not just the
    calculation.
-   The first run can be affected by warm-up, system load, and other
    environmental factors.
-   The notebook does not verify that all three outputs match.
-   If compilation fails, the original code may still attempt to run the
    class and report a misleading time.

For a stronger comparison, check successful compilation, capture and
compare the outputs, run multiple warm-up and timed iterations, and
compare average or median times under the same conditions.

------------------------------------------------------------------------

# 14. End-to-End Pipeline

``` text
API Keys from Colab Secrets
          ↓
Gemini and Groq Clients
          ↓
Model IDs
          ↓
System Prompt + Python Benchmark
          ↓
messages_for()
          ↓
port(client, model, python_code)
          ↓
Three Generated Java Source Strings
          ↓
Display Generated Code
          ↓
Save Java Files
          ↓
javac Compilation
          ↓
java Execution
          ↓
Elapsed-Time Measurements
```

------------------------------------------------------------------------

# 15. Key Concepts

  ---------------------------------------------------------------------------
  Concept                                 Meaning
  --------------------------------------- -----------------------------------
  API key                                 Secret used to authenticate an API
                                          request

  Base URL                                API endpoint the client sends
                                          requests to

  Model ID                                Identifier for the model requested
                                          from a provider

  OpenAI-compatible API                   API interface that follows a
                                          compatible request format

  System prompt                           Instructions that define the
                                          model's role and expected behavior

  User message                            The task and code sent to the model

  `messages_for()`                        Builds the chat messages for a
                                          Python program

  `port()`                                Sends a translation request and
                                          extracts the generated Java code

  `chat.completions.create()`             Makes a chat-completion API request

  `response.choices[0].message.content`   Accesses the first response's
                                          message text

  `Markdown`                              Formats output for display in a
                                          notebook

  `subprocess.run()`                      Runs an external program from
                                          Python

  `javac`                                 Compiles Java source code

  `java`                                  Launches a compiled Java program

  `time.time()`                           Returns the current wall-clock time

  Execution-time benchmark                Measures how long a program takes
                                          to run
  ---------------------------------------------------------------------------

------------------------------------------------------------------------

# 16. Key Takeaways

1.  LLMs can be used to translate code from one programming language to
    another.
2.  A system prompt sets expectations for the generated Java code.
3.  Chat APIs receive messages with roles such as `system` and `user`.
4.  `messages_for()` separates message preparation from API execution.
5.  `port()` makes it easy to reuse the same translation process for
    different clients and models.
6.  A model variable's name does not determine which provider receives
    the request; the client does.
7.  Generated code should be inspected, compiled, and tested instead of
    assumed to be correct.
8.  `javac` compiles source code, while `java` runs the compiled class.
9.  Runtime measurements are meaningful only when the implementations
    produce equivalent results and the test conditions are fair.
10. The notebook's timing section measures each generated program once,
    so repeated trials would improve the comparison.

------------------------------------------------------------------------

# 17. Interview Questions

### Q1. What is the goal of this project?

**Answer:** The goal is to use LLMs to translate a Python benchmark into
Java and compare the generated implementations by inspecting the code,
compiling it, and measuring execution time.

### Q2. Why do we use a system prompt?

**Answer:** The system prompt tells the model what task to perform and
how to format its response. Here, it asks for high-performance Java code
without explanations or Markdown formatting.

### Q3. Why is `messages_for()` a separate function?

**Answer:** It builds the standard message list in one place. We can
reuse the same system instructions with different Python programs
without duplicating the message structure.

### Q4. What does the `port()` function do?

**Answer:** It accepts a client, model ID, and Python code; prepares the
messages; calls the chat-completions API; extracts the generated Java
code; removes common Markdown fences; and returns the result.

### Q5. What is the difference between a client and a model ID?

**Answer:** The client contains the provider connection settings, such
as the API key and base URL. The model ID tells that provider which
model to use for a request.

### Q6. Why can the same `OpenAI` client class be used for Gemini and Groq?

**Answer:** Both services expose OpenAI-compatible API endpoints. The
Python client can use their compatible interface when configured with
the appropriate base URL and API key.

### Q7. Does naming a variable `openai_java` mean the request was sent to OpenAI?

**Answer:** No. The request destination is determined by the client
passed to `port()`. In this notebook, `port(groq, OPENAI_MODEL, ...)`
uses the Groq client.

### Q8. What is the purpose of `subprocess.run()`?

**Answer:** It allows Python to execute external commands. In this
project, it runs the Java compiler and launches the compiled Java
programs.

### Q9. What is the difference between `javac` and `java`?

**Answer:** `javac` compiles `.java` source files into bytecode. `java`
runs the compiled class using the JVM.

### Q10. How does the notebook measure execution time?

**Answer:** It records the time before launching the Java process and
subtracts that start time from the time after the process finishes.

### Q11. Does this benchmark prove which model is best?

**Answer:** Not by itself. A faster result does not prove correctness,
and the current notebook runs each program only once. We should verify
output equivalence and repeat the measurements under consistent
conditions.

### Q12. Why should we compare the numerical outputs?

**Answer:** Two programs should perform the same task before their speed
can be compared fairly. If one generated implementation calculates a
different result, its runtime is not a valid like-for-like comparison.
