# Day 15: Meeting Minutes Generator — Audio Transcription & LLM Summarization

## Overview

Day 15 builds an end-to-end AI product that takes an audio recording of a meeting and produces structured meeting minutes.

The project combines:

- **Audio transcription** (speech → text)
- **LLM summarization** (text → structured minutes)

The overall pipeline is:

```text
Audio File (.mp3)
      ↓
Transcription (Whisper or OpenAI)
      ↓
Raw Text Transcript
      ↓
LLM (Llama 3.2)
      ↓
Formatted Meeting Minutes (Markdown)
```

---

## 🎯 Learning Objectives

- Understand the two-step pipeline: transcription → summarization.
- Transcribe audio using an open-source Whisper model through Hugging Face Pipelines.
- Transcribe audio using the OpenAI API as an alternative.
- Compare open-source and API-based transcription.
- Mount Google Drive in Colab to access audio files.
- Use `BitsAndBytesConfig` for 4-bit quantization.
- Use the low-level Transformers API to generate meeting minutes with Llama 3.2.
- Stream generated output using `TextStreamer`.
- Display formatted Markdown output in a notebook.

---

# 1. Project Architecture

The project has two main steps:

```text
STEP 1: Transcribe Audio
           ↓
     Raw text transcript

STEP 2: Analyze & Report
           ↓
     Structured meeting minutes
```

### Step 1 — Transcription

Converts spoken audio into text using either:

- **Option A:** Open-source Whisper model via Hugging Face `pipeline()`
- **Option B:** OpenAI Audio Transcription API

### Step 2 — Summarization

Sends the transcript to Llama 3.2 with a system prompt requesting structured meeting minutes.

---

# 2. Audio Source

The notebook uses a portion of **Denver City Council meeting minutes**.

The audio file is stored on Google Drive and accessed by mounting the drive in Colab:

```python
from google.colab import drive

drive.mount("/content/drive")

audio_filename = "/content/drive/MyDrive/llms/denver_extract.mp3"
```

### `drive.mount()`

Connects the Colab runtime to your Google Drive so files on the drive can be accessed like local files.

---

# 3. Install Required Libraries

```python
!pip install -q --upgrade bitsandbytes accelerate transformers==4.57.6
```

### Libraries

| Library | Purpose |
|---|---|
| `transformers` | Hugging Face models, tokenizers, and pipelines |
| `accelerate` | Efficient CPU/GPU model loading |
| `bitsandbytes` | Model quantization |
| `openai` | OpenAI API client |
| `torch` | PyTorch deep-learning framework |

---

# 4. Imports

```python
import os
import requests
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from google.colab import drive
from huggingface_hub import login
from google.colab import userdata
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig,
)
import torch
```

### Important APIs

- `AutoTokenizer` → loads the model's tokenizer.
- `AutoModelForCausalLM` → loads a causal language model.
- `TextStreamer` → streams generated text token by token.
- `BitsAndBytesConfig` → configures quantization.
- `pipeline` → high-level Hugging Face API for tasks like speech recognition.
- `OpenAI` → client for the OpenAI API.
- `Markdown`, `display` → renders formatted output in the notebook.

---

# 5. Authentication

### Hugging Face Login

```python
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
```

The token is stored in Colab Secrets.

### OpenAI API Key

```python
openai_api_key = userdata.get('OPENAI_API_KEY')
openai = OpenAI(api_key=openai_api_key)
```

---

# 6. Model Used

```python
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
```

Llama 3.2 3B Instruct is used for the summarization step.

| Feature | Value |
|---|---|
| Model | Llama-3.2-3B-Instruct |
| Company | Meta |
| Parameters | 3B |
| Type | Instruction-tuned |

---

# 7. STEP 1 — Transcribe Audio

## 7.1 Option 1: Open-Source Transcription with Whisper

### What is Whisper?

**Definition:**

> Whisper is an open-source automatic speech recognition (ASR) model developed by OpenAI, available through Hugging Face, that converts spoken audio into text.

**Think:**
**Whisper = Speech → Text**

### Code

```python
from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium.en",
    dtype=torch.float16,
    device='cuda',
    return_timestamps=True
)

result = pipe(audio_filename)
transcription = result["text"]
print(transcription)
```

### Code breakdown

#### `pipeline("automatic-speech-recognition")`

Creates a high-level speech recognition pipeline.

#### `model="openai/whisper-medium.en"`

Uses the Whisper medium English model from Hugging Face Hub.

#### `dtype=torch.float16`

Loads the model in half precision to reduce GPU memory usage.

#### `device='cuda'`

Runs inference on the GPU.

#### `return_timestamps=True`

Enables timestamp information during transcription, which helps with processing longer audio files.

#### `result["text"]`

The pipeline returns a dictionary. The transcribed text is in the `"text"` key.

### Conceptual flow

```text
Audio File (.mp3)
      ↓
Whisper Pipeline
      ↓
result["text"]
      ↓
Raw Text Transcript
```

### Saving the open-source result

```python
open_source_transcription = transcription
```

This stores the open-source transcription separately so it can be compared with the API-based transcription later.

---

## 7.2 Option 2: OpenAI API Transcription

```python
AUDIO_MODEL = "gpt-4o-mini-transcribe"

openai_api_key = userdata.get('OPENAI_API_KEY')
openai = OpenAI(api_key=openai_api_key)

transcription = openai.audio.transcriptions.create(
    model=AUDIO_MODEL,
    file=audio_file,
    response_format="text"
)
print(transcription)
```

### Code breakdown

#### `AUDIO_MODEL = "gpt-4o-mini-transcribe"`

The OpenAI transcription model being used.

#### `openai.audio.transcriptions.create()`

Sends the audio file to OpenAI's API for transcription.

#### `response_format="text"`

Returns the transcription as plain text.

---

## 7.3 Comparing Both Transcriptions

```python
display(Markdown(open_source_transcription))
print("\n\n")
display(Markdown(transcription))
```

This displays both transcriptions side-by-side for comparison.

| Feature | Open-Source (Whisper) | OpenAI API |
|---|---|---|
| Cost | Free (GPU required) | Pay per request |
| Privacy | Audio stays local | Audio sent to API |
| Control | Full model control | API-dependent |
| Model | `openai/whisper-medium.en` | `gpt-4o-mini-transcribe` |
| Setup | Hugging Face + GPU | API key |

---

# 8. STEP 2 — Analyze & Report

## 8.1 Define the Prompt

The notebook constructs a system message and a user prompt:

```python
system_message = """
You produce minutes of meetings from transcripts, with summary,
key discussion points, takeaways and action items with owners,
in markdown format without code blocks.
"""

user_prompt = f"""
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners

Transcription:
{transcription}
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]
```

### System message

Tells the LLM its role: produce meeting minutes from a transcript in a specific format.

### User prompt

Provides the actual transcript and requests structured output including:

- Summary with attendees, location, and date
- Discussion points
- Takeaways
- Action items with owners

### Messages structure

The standard chat-message format used by instruction-tuned models:

```text
System Message → Sets the role/behavior
       +
User Message   → Contains the transcript and instructions
       ↓
LLM generates meeting minutes
```

---

## 8.2 4-Bit Quantization

```python
quant_config = BitsAndBytesConfig(
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

This is the same quantization approach used in Day 14, applied here so the 3B model fits on a Colab T4 GPU.

---

## 8.3 Load Tokenizer, Apply Chat Template & Load Model

```python
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt"
).to("cuda")

streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map="auto",
    quantization_config=quant_config
)
```

### Code breakdown

#### Load tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
```

Loads the tokenizer associated with Llama 3.2. Sets the pad token to prevent padding warnings.

#### Apply chat template

```python
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt"
).to("cuda")
```

Converts the messages into the Llama-specific conversation format, returns PyTorch tensors, and moves them to the GPU.

#### TextStreamer

```python
streamer = TextStreamer(tokenizer)
```

`TextStreamer` displays generated tokens in real time as they are produced, instead of waiting for the entire generation to complete.

#### Load model

```python
model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map="auto",
    quantization_config=quant_config
)
```

Loads the Llama 3.2 model with automatic device mapping and 4-bit quantization.

### Complete flow

```text
Messages
   ↓
AutoTokenizer
   ↓
apply_chat_template()
   ↓
Token IDs / Tensor → GPU
   ↓
AutoModelForCausalLM (4-bit quantized)
   ↓
Ready for generation
```

---

## 8.4 Generate Meeting Minutes

```python
outputs = model.generate(
    inputs,
    max_new_tokens=2000,
    streamer=streamer
)
```

### `max_new_tokens=2000`

Meeting minutes can be lengthy, so the model is allowed to generate up to 2000 new tokens.

### `streamer=streamer`

Passes the `TextStreamer` so the generated text is printed token by token as it is produced.

---

## 8.5 Decode & Display the Output

```python
response = tokenizer.decode(outputs[0])

display(Markdown(response))
```

### `tokenizer.decode(outputs[0])`

Converts the generated token IDs back into readable text.

### `display(Markdown(response))`

Renders the response as formatted Markdown in the notebook, showing the meeting minutes with proper headings, bullet points, and structure.

---

# 9. End-to-End Pipeline

This is the most important diagram from Day 15:

```text
Audio File (.mp3)
       ↓
  ┌────┴────┐
  │         │
Whisper   OpenAI API
  │         │
  └────┬────┘
       ↓
Raw Text Transcript
       ↓
System + User Prompt
       ↓
AutoTokenizer
       ↓
apply_chat_template()
       ↓
Token IDs → GPU
       ↓
Llama 3.2 (4-bit quantized)
       ↓
model.generate(max_new_tokens=2000)
       ↓
tokenizer.decode()
       ↓
Formatted Meeting Minutes (Markdown)
```

---

# 10. Colab Pro-Tip: CUDA Runtime Error

When running in Google Colab, you may encounter:

```text
Runtime error: CUDA is required but not available for bitsandbytes.
```

This error is **misleading**. It usually means Google has switched out your Colab runtime.

### Solution

```text
1. Kernel menu >> Disconnect and delete runtime
2. Reload the colab and Edit menu >> Clear All Outputs
3. Connect to a new T4 using the button at the top right
4. Select "View resources" to confirm GPU availability
5. Rerun the cells from the top
```

Do **not** try changing package versions — a fresh runtime is the fix.

---

# 11. Key Concepts

| Concept | Meaning |
|---|---|
| Whisper | Open-source speech recognition model |
| `pipeline("automatic-speech-recognition")` | High-level ASR pipeline |
| `openai/whisper-medium.en` | Whisper medium English model |
| `return_timestamps=True` | Enables timestamps during transcription |
| `openai.audio.transcriptions.create()` | OpenAI API transcription |
| `BitsAndBytesConfig` | Configures 4-bit quantization |
| `AutoTokenizer` | Loads the model's tokenizer |
| `AutoModelForCausalLM` | Loads a causal language model |
| `apply_chat_template()` | Formats chat messages for a model |
| `model.generate()` | Generates new tokens |
| `TextStreamer` | Streams generated text in real time |
| `tokenizer.decode()` | Converts token IDs to text |
| `display(Markdown(...))` | Renders Markdown in Colab |
| `drive.mount()` | Connects Google Drive to Colab |

---

# 12. Key Takeaways

1. An AI product can chain multiple models: one for transcription, another for summarization.
2. Whisper provides free, open-source audio transcription through Hugging Face Pipelines.
3. OpenAI's API provides an alternative transcription option that is simpler to set up but costs money and sends data externally.
4. The two approaches can be compared side-by-side.
5. System and user prompts guide the LLM to produce structured output.
6. 4-bit quantization allows larger models to fit on limited GPU hardware.
7. `TextStreamer` provides real-time output during generation.
8. `max_new_tokens=2000` is needed because meeting minutes can be long.
9. `display(Markdown(...))` renders the final output with proper formatting.
10. Google Drive can be mounted in Colab to access audio files.
11. Misleading CUDA errors in Colab are often solved by reconnecting to a fresh runtime.

---

# 13. Interview Questions

### Q1. What is Whisper?

**Answer:**  
Whisper is an open-source automatic speech recognition model developed by OpenAI that converts spoken audio into text. It is available through Hugging Face.

### Q2. What are the two transcription options used in this project?

**Answer:**  
Option 1 uses the open-source Whisper model via Hugging Face `pipeline()`. Option 2 uses the OpenAI Audio Transcription API.

### Q3. Why is `return_timestamps=True` used in the Whisper pipeline?

**Answer:**  
It enables timestamp information during transcription, which helps the model process longer audio files correctly.

### Q4. What does the system message do in this project?

**Answer:**  
The system message tells the LLM its role: to produce structured meeting minutes from a transcript, including a summary, discussion points, takeaways, and action items.

### Q5. Why is `max_new_tokens` set to 2000?

**Answer:**  
Meeting minutes can be lengthy. A higher token limit allows the model to generate a complete, detailed summary without cutting off.

### Q6. What is `TextStreamer`?

**Answer:**  
`TextStreamer` is a Hugging Face utility that displays generated tokens in real time as they are produced, providing immediate feedback during text generation.

### Q7. Why is 4-bit quantization used?

**Answer:**  
4-bit quantization reduces the memory required to load the model, allowing the 3B-parameter Llama model to fit on a Colab T4 GPU with limited VRAM.

### Q8. What does `drive.mount()` do?

**Answer:**  
It connects the Google Colab runtime to Google Drive, allowing the notebook to read files stored on the drive as if they were local files.
