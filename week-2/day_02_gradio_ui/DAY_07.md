# Day 7: Outrageously Simple UIs with Gradio

## 📋 Overview

Day 7 introduces **Gradio**, a Python-based framework for building interactive web interfaces for machine learning models and Python functions.

Gradio allows us to create web applications without manually writing HTML, CSS, or JavaScript.

In this lesson, we will learn how to:

- Create a simple web interface using `gr.Interface()`.
- Understand how Gradio connects Python functions to UI components.
- Build a chatbot interface using `gr.ChatInterface()`.
- Connect a chatbot to the Groq API.
- Understand how Gradio provides chat history.
- Clean and convert Gradio messages into the format expected by Groq.
- Launch a Gradio application locally.

---

## 🎯 Learning Objectives

By the end of this lesson, you should be able to:

1. Understand the basic working of Gradio.
2. Connect Python functions with UI components.
3. Create a text-processing web application.
4. Understand the `message` and `history` parameters of `gr.ChatInterface()`.
5. Connect Gradio with the Groq API.
6. Understand why Gradio history may require conversion before sending it to Groq.
7. Use `.launch()` to run a Gradio application.
8. Explain the difference between a warning and an API error.

---

# 📖 Key Concepts Explained

## 1. What is Gradio?

**Gradio** is a Python library that allows us to create interactive web interfaces for Python functions and machine learning models.

Normally, creating a web application may require:

- HTML for structure.
- CSS for styling.
- JavaScript for interactions.
- Backend code to process user requests.

Gradio simplifies this process by allowing us to build interfaces directly using Python.

### Example

```python
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(
    fn=greet,
    inputs="textbox",
    outputs="textbox"
)

demo.launch()
```

When we run this code, Gradio creates a web interface where users can enter their names and receive a greeting.

---

## 2. How Does Gradio Work?

Gradio connects three main components:

1. **Input:** Data entered by the user.
2. **Python function:** Processes the input.
3. **Output:** Displays the function's return value.

### Workflow

```text
User enters input
        |
        v
Gradio receives input
        |
        v
Python function executes
        |
        v
Function returns result
        |
        v
Gradio displays output
```

### Important Parameters

| Parameter | Meaning |
|---|---|
| `fn` | Python function to execute |
| `inputs` | UI component that receives user input |
| `outputs` | UI component that displays the result |
| `launch()` | Starts the Gradio application |

---

# 💻 Code Walkthrough

## 3. Importing Gradio

In our notebook, we first check whether Gradio is installed.

```python
import sys
import subprocess

try:
    import gradio as gr
    print("Gradio is already installed.")

except ImportError:
    print("Gradio not found. Installing now...")

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "gradio"
    ])

    import gradio as gr

print("All imports successful!")
```

### Explanation

- `sys.executable` identifies the Python interpreter currently running.
- `subprocess.check_call()` executes the installation command.
- `import gradio as gr` imports Gradio using the shorter name `gr`.
- `try` and `except` allow us to handle the case where Gradio is not installed.

**Note:** If you install Gradio in a running Jupyter environment, you may need to restart the kernel if the environment has changed.

---

## 4. Creating a Basic Text Shouter App

In this example, we create a Python function that converts text into uppercase.

### Python Function

```python
def shout(text):
    print(f"Python function received input: '{text}'")
    return text.upper()
```

### Explanation

- `text` receives the user's input.
- `text.upper()` converts all lowercase letters to uppercase.
- `return` sends the result back to Gradio.

### Creating the Interface

```python
demo = gr.Interface(
    fn=shout,
    inputs="textbox",
    outputs="textbox",
    flagging_mode="never"
)
```

### Explanation

| Code | Purpose |
|---|---|
| `fn=shout` | Connects the Python function |
| `inputs="textbox"` | Creates a text input |
| `outputs="textbox"` | Displays the returned text |
| `flagging_mode="never"` | Disables the feedback flagging feature |

### Launching the App

```python
demo.launch()
```

This starts the Gradio web application.

### Example

Input:

```text
hello arpan
```

Output:

```text
HELLO ARPAN
```

### Key Takeaway

Gradio automatically passes the input value to the Python function and displays the function's return value.

---

# 🤖 Building an AI Chatbot with Gradio and Groq

## 5. What is `gr.ChatInterface()`?

`gr.ChatInterface()` is a Gradio component designed specifically for building chatbot applications.

It provides a chat interface and manages the conversation interaction.

Instead of creating a textbox and manually managing every chat submission, we can connect a Python function to `gr.ChatInterface()`.

### Basic Structure

```python
def chatbot_function(message, history):
    # Process the message
    return "AI response"

gr.ChatInterface(
    fn=chatbot_function
).launch()
```

The function receives two important arguments.

| Parameter | Meaning |
|---|---|
| `message` | The latest user message |
| `history` | Previous conversation messages |

---

## 6. Understanding `message` and `history`

Suppose a user has this conversation:

```text
User: Hello
Assistant: Hi!

User: How are you?
```

When the user sends the second message, the function receives:

```python
message = "How are you?"
```

The history contains the previous conversation.

Conceptually, it may look like:

```python
history = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
]
```

The exact structure depends on the Gradio version and the configured chatbot message format.

**Important:** Do not assume that every Gradio history entry is already in the exact format required by an external LLM API.

---

# 7. Connecting Gradio to Groq

In our notebook, we use the Groq API to generate chatbot responses.

Groq provides an API that can be accessed through its Python SDK.

### Required Libraries

```python
from dotenv import load_dotenv
import os
from groq import Groq
```

### Loading Environment Variables

```python
load_dotenv()
```

This loads environment variables from a `.env` file, if available.

For example, your `.env` file may contain:

```text
GROQ_API_KEY=your_groq_api_key
```

Keep your API key private. Do not upload it to GitHub or share it publicly.

### Creating the Groq Client

```python
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
```

`os.getenv("GROQ_API_KEY")` retrieves the API key from the environment.

The `Groq()` client is then used to send requests to the Groq API.

---

# 8. Understanding the Chatbot Message Structure

Before sending a request to Groq, we need to understand how chat messages are represented.

A typical chat completion message contains:

```python
{
    "role": "user",
    "content": "Hello"
}
```

### Meaning of the Fields

| Field | Meaning |
|---|---|
| `role` | Identifies who sent the message |
| `content` | Contains the message text |

Common roles include:

- `system`: Instructions that guide the assistant.
- `user`: A message from the user.
- `assistant`: A message generated by the AI.

### Example Conversation

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Hello"
    },
    {
        "role": "assistant",
        "content": "Hi! How can I help?"
    }
]
```

This list represents the conversation context sent to the model.

---

# 9. Why Do We Clean Gradio Messages?

During our chatbot implementation, we encountered this error:

```text
groq.BadRequestError: Error code: 400

' messages.1 ' :
property 'metadata' is unsupported
```

The problem was that the messages received from Gradio contained additional fields that Groq did not accept.

### Actual Gradio History from Our Notebook

We inspected the history and received a structure similar to this:

```python
[
    {
        "role": "user",
        "metadata": None,
        "content": [
            {
                "text": "Hello, My name Arpan",
                "type": "text"
            }
        ],
        "options": None
    }
]
```

Notice the differences from the simpler chat completion message.

### Gradio History vs. Groq Message

| Gradio history | Groq message |
|---|---|
| `role` | `role` |
| `content` may be a list of content blocks | `content` should be a supported string or content format |
| May include `metadata` | Unsupported extra field |
| May include `options` | Unsupported extra field |

### Where Does Metadata Come From?

We did not manually add `metadata` in our chatbot function.

It was already present in the history supplied to our function by Gradio.

Gradio's message format can include fields such as `metadata` and `options`.

Our code copied those history entries into the messages list using:

```python
messages.extend(history)
```

Therefore, the extra fields were also copied into the request we sent to Groq.

**Important:** The exact source of those fields depends on the Gradio version and configuration. Our printed history confirms that they were present in the history received by our function.

---

# 10. Cleaning and Converting Gradio Messages

We need to convert Gradio's message structure into a format that Groq can accept.

The conversion has two main tasks:

1. Keep only the required `role` and `content` fields.
2. Convert text-based content blocks into a string.

### Corrected Code

```python
clean_messages = []

for msg in messages:
    content = msg["content"]

    # Convert Gradio's list-based text content into a string
    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if item.get("type") == "text"
        )

    clean_messages.append({
        "role": msg["role"],
        "content": content
    })
```

### Explanation

#### Step 1: Create an Empty List

```python
clean_messages = []
```

We create a new list to store the cleaned messages.

#### Step 2: Loop Through Messages

```python
for msg in messages:
```

We process each message individually.

#### Step 3: Extract the Content

```python
content = msg["content"]
```

We retrieve the content of the current message.

#### Step 4: Check Whether Content Is a List

```python
if isinstance(content, list):
```

If the content is a list, we convert its text blocks into a string.

#### Step 5: Keep Only the Required Fields

```python
clean_messages.append({
    "role": msg["role"],
    "content": content
})
```

This creates a new message containing only the `role` and `content` fields.

### Example Conversion

Before:

```python
{
    "role": "user",
    "metadata": None,
    "content": [
        {"text": "Hello Arpan", "type": "text"}
    ],
    "options": None
}
```

After:

```python
{
    "role": "user",
    "content": "Hello Arpan"
}
```

### Important Note

This conversion is suitable for the text-only history format shown in our notebook.

If the chatbot supports images, audio, or other multimodal content, those content blocks require appropriate handling rather than simply discarding them.

---

# 11. Understanding `list.extend()` and `list.append()`

Our chatbot uses two important Python list methods.

### `extend()`

```python
messages.extend(history)
```

Adds the elements of another iterable to the existing list.

Example:

```python
messages = ["system"]
history = ["user", "assistant"]

messages.extend(history)

print(messages)
```

Output:

```python
["system", "user", "assistant"]
```

### `append()`

```python
messages.append({
    "role": "user",
    "content": message
})
```

Adds one new element to the list.

Example:

```python
messages = ["system"]

messages.append("user")

print(messages)
```

Output:

```python
["system", "user"]
```

### Difference

| Method | What it adds |
|---|---|
| `append()` | One object as a single element |
| `extend()` | Elements from an iterable |

---

# 12. Complete Groq Chatbot Code

This is the corrected chatbot implementation based on our notebook.

```python
from dotenv import load_dotenv
import os
import gradio as gr
from groq import Groq

# Load environment variables
load_dotenv()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# Define chatbot function
def local_chatbot(message, history):

    # System instruction
    messages = [
        {
            "role": "system",
            "content": "You are a friendly, helpful local AI assistant"
        }
    ]

    # Add previous conversation
    messages.extend(history)

    # Add the latest user message
    messages.append({
        "role": "user",
        "content": message
    })

    # Gradio history may contain extra fields and list-based
    # content, so convert it to Groq's expected message format.
    clean_messages = []

    for msg in messages:
        content = msg["content"]

        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if item.get("type") == "text"
            )

        clean_messages.append({
            "role": msg["role"],
            "content": content
        })

    # Send request to Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=clean_messages
    )

    # Return AI response
    return response.choices[0].message.content


# Create chatbot interface
chatbot_ui = gr.ChatInterface(
    fn=local_chatbot,
    title="Arpan Chatbot",
    description="Ask anything! Powered by Groq API and Gradio."
)

# Launch chatbot
chatbot_ui.launch()
```

### What Happens When We Send a Message?

```text
User types a message
        |
        v
Gradio calls local_chatbot()
        |
        v
Function receives message and history
        |
        v
System instruction is added
        |
        v
Previous history is added
        |
        v
Current user message is added
        |
        v
Messages are cleaned and converted
        |
        v
Request is sent to Groq
        |
        v
Groq generates the response
        |
        v
Response text is returned to Gradio
        |
        v
Chatbot displays the response
```

---

# 13. Understanding the Groq API Response

Our API request is:

```python
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=clean_messages
)
```

### Parameters

| Parameter | Meaning |
|---|---|
| `model` | The model we want to use |
| `messages` | The conversation sent to the model |

The response contains the generated assistant message.

We retrieve its text using:

```python
response.choices[0].message.content
```

### Breaking It Down

- `response`: The API response object.
- `choices`: The available completion choices.
- `[0]`: Selects the first choice.
- `message`: The generated assistant message.
- `content`: The text of that message.

We return this text to Gradio:

```python
return response.choices[0].message.content
```

---

# 14. Understanding `.launch()`

The `.launch()` method starts the Gradio application.

```python
chatbot_ui.launch()
```

Gradio runs a local web server and provides a URL where we can interact with the application.

A common local address is:

```text
http://127.0.0.1:7860
```

The exact port may differ if the default port is already in use.

### Local vs. Public Sharing

```python
chatbot_ui.launch()
```

Launches the interface locally by default.

```python
chatbot_ui.launch(share=True)
```

Requests a shareable public URL.

**Security note:** A public link may allow others to access your application. Avoid exposing private information or sensitive functionality.

---

# 15. Understanding the Starlette Deprecation Warning

During our chatbot execution, we also saw:

```text
StarletteDeprecationWarning:

'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated.

Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead.
```

### What Does It Mean?

A deprecation warning means that a library is using an older name or feature that is being phased out.

In our case, the warning concerns a status-code constant used by the Gradio/Starlette stack.

It is separate from the Groq API error.

### Difference Between the Two Messages

| Message | Meaning |
|---|---|
| `StarletteDeprecationWarning` | A deprecated library constant is being used |
| `groq.BadRequestError: 400` | Groq rejected the API request |

The warning is not itself the reason the Groq request failed.

---

# ❓ Interview Questions & Answers

## Q1. What is Gradio?

**Answer:** Gradio is a Python library used to create interactive web interfaces for Python functions and machine learning models with minimal frontend code.

## Q2. What is the purpose of `gr.Interface()`?

**Answer:** `gr.Interface()` connects a Python function to input and output UI components, allowing users to interact with the function through a web interface.

## Q3. What is the purpose of `gr.ChatInterface()`?

**Answer:** `gr.ChatInterface()` provides a dedicated chatbot interface and passes the current message and conversation history to a Python function.

## Q4. What are `message` and `history` in Gradio?

**Answer:**

- `message` contains the latest user input.
- `history` contains the previous conversation, represented according to the configured Gradio message format.

## Q5. What is the difference between `append()` and `extend()`?

**Answer:** `append()` adds one object to a list, while `extend()` adds the elements of an iterable to the list.

## Q6. Why did we use `clean_messages` in our Groq chatbot?

**Answer:** Our Gradio history contained extra fields such as `metadata` and `options`, and its text content was represented as a list of dictionaries. We converted the history to a simpler format containing supported roles and text content before sending it to Groq.

## Q7. Where did the metadata in our chatbot come from?

**Answer:** It was present in the history supplied to our chatbot function by Gradio. We confirmed this by printing the history. Our code copied the history into the API message list, which is why the extra field reached Groq.

## Q8. What does `load_dotenv()` do?

**Answer:** It loads environment variables from a `.env` file into the process environment, allowing us to access configuration values such as API keys.

## Q9. What does `.launch()` do?

**Answer:** It starts the Gradio application and makes the interface accessible through a local web address or a configured sharing URL.

## Q10. What is the difference between a warning and an exception?

**Answer:** A warning indicates something that may need attention, such as deprecated functionality. An exception represents an error that interrupts normal execution unless handled.



## Summary

In this lesson, we learned how to use Gradio to create interactive web applications using Python.

We started with a simple text-shouting app and then built a chatbot connected to the Groq API.

The most important practical lesson from our debugging experience was:

**The message format provided by a UI framework may not be identical to the message format expected by an external API.**

By inspecting the actual history and converting it into a supported format, we made our chatbot integration more reliable.
