# 🗺️ LLM Engineering Journey — Learning Roadmap

This roadmap tracks my progress through Ed Donner's **"LLM Engineering: Master AI and Large Language Models"** course. 

I am documenting and building every day module from scratch using a fully local setup (Ollama).

---

## 📈 Current Progress

- **Legend:** ✅ Complete | 🔄 In Progress | ⏳ Pending

### Week 1 — Foundations of LLM Engineering (Complete)
*Intuitively understand local inference, message structures, prompt design, and tokenization.*

- [x] **Day 1: Hello, LLMs! — First Contact with Ollama**
  - [x] Set up local Ollama with `llama3.2:1b` and `gemma3:270m`
  - [x] Build simple chat completion API call
  - [x] Learn system, user, and assistant roles
  - [x] Write step-by-step chat memory history manager
  - [x] Benchmark speed and latency differences between models
- [x] **Day 2: LLM API Parameters & Prompt Designs**
  - [x] Master temperature (0.0 vs 1.0) and observe random outputs
  - [x] Pull data in valid JSON format using native format controls
  - [x] Create reusable translations using prompt templates
- [x] **Day 3: Prompt Engineering Mastery**
  - [x] Write few-shot classifier prompt to output single emojis
  - [x] Implement Chain of Thought reasoning for math puzzles
  - [x] Build an AI Code Reviewer system to grade and optimize code
- [x] **Day 4: Tokens, Context Windows, and LLM Internals**
  - [x] Tokenize sentences and decode token IDs using `tiktoken`
  - [x] Budget context window limits and build safety checks
  - [x] Run letter count tests to explore tokenizer limitations
- [/] **Day 5 — AI Brochure Generator (Week 1 Capstone)** 🔄
  - [/] Build `Website` scraper class with BeautifulSoup (in `scraper.py`)
  - [/] Create brochure generation pipeline (in notebook)
  - [ ] Add multi-model support
  - [ ] Build Gradio web UI with streaming
  - [ ] Write DAY_05.md documentation

---

## 🔮 Future Modules (Coming Soon)

### Week 2 — Frontier APIs & Chatbots (Next Up)
- [ ] **Day 5: Web Scraping & Capstone Brochure Generator**
- [ ] **Day 6: Function Calling & OpenAI Tools**
- [ ] **Day 7: Multi-Modal Chatbots (Gradio UI + Audio/Images)**
- [ ] **Day 8: Customer Support Agents with Memory**

### Week 3 — Audio & Open-Source Models
- [ ] **Meeting Minutes Generator** (Whisper audio transcription)

### Week 4 — Code Generation & Optimization
- [ ] **AI Code Optimizer (Python → C++)**

### Week 5 — Retrieval-Augmented Generation (RAG)
- [ ] **RAG Knowledge Worker** (Vector databases & semantic search)

### Week 6 — Model Fine-Tuning (Part 1)
- [ ] **Price Predictor Application**

### Week 7 — Model Fine-Tuning (Part 2)
- [ ] **Fine-Tuning LLMs using QLoRA**

### Week 8 — Multi-Agent Systems
- [ ] **Multi-Agent Deal Spotter**
