# Day 8: Gradio Web UI for the AI Brochure Generator

## 📋 Overview
Day 8 is the second Capstone-level project. We take our modular scraping, link filtering, and text-generation logic from Week 1 and bring it into the web browser. Instead of using basic side-by-side textbox widgets, we learn how to use Gradio's advanced layout engine, **`gr.Blocks()`**, to build a customized, centered dashboard. We also learn how to wire Python generator functions (`yield`) to the UI, enabling the website brochure to stream dynamically in markdown directly below the input fields.

---

## 🎯 Learning Objectives
* Master Gradio's layout engine (`gr.Blocks`, `gr.Row`, `gr.Column`) to build custom user interfaces.
* Implement centering and column scaling for clean dashboard layouts.
* Wire multi-stage scraping and filtering logic to progressive progress yields.
* Stream text output to Gradio Markdown panels in real-time.

---

## 📖 Key Concepts Explained (Beginner-Friendly)

### 1. High-Level Layouts: gr.Blocks() vs gr.Interface()
* **`gr.Interface`:** The basic constructor. It automatically places all inputs on the left and outputs on the right. While fast, you cannot change the layout, add text dividers, or position buttons.
* **`gr.Blocks`:** The advanced layout builder. It acts as a blank grid where you can structure components vertically (stacked) or horizontally (side-by-side) using rows and columns.

### 2. Centering Elements in Gradio Grid
By default, block layouts fill the width of the screen. To center inputs (like a search bar or model selector):
* We create a Row: `with gr.Row():`
* Inside the row, we define three Columns.
* We set `scale` parameters on the columns (e.g. `scale=1` on the outer spacers, and `scale=2` on the center column). 
* Gradio will automatically center your input controls by allocating 25% width to the left spacer, 50% width to the inputs, and 25% width to the right spacer.

### 3. Pushing Live Text to Gradio
If a function returns a standard string using `return`, the UI stays blank during the entire scraping and generation process (which can take 15–30 seconds), making it look like the app crashed.

Gradio natively supports Python generators. When a function uses the **`yield`** keyword:
* The function executes until it reaches a `yield` statement.
* It pauses and sends that intermediate text to the UI immediately.
* When the UI displays it, the function resumes.
* This allows us to display progress notifications (e.g., *"Step 1: Scraping homepage..."*) first, and then transition smoothly into streaming the sales brochure word-by-word.

---

## 💻 Code Walkthrough (Simplified)

### 1. Centered Blocks Layout
```python
import gradio as gr

def process_data(user_input):
    yield "Working..."
    # Processing logic
    yield "Done!"

with gr.Blocks() as demo:
    gr.Markdown("# <center>My App</center>")
    
    with gr.Row():
        
        with gr.Column(scale=2): # Centered Inputs
            inp = gr.Textbox(label="Enter value")
            btn = gr.Button("Submit", variant="primary")
            
        
    with gr.Row():
        with gr.Column(scale=4): # Wide Output below
            out = gr.Markdown()
        
    btn.click(fn=process_data, inputs=inp, outputs=out)

demo.launch()
```

---

## ❓ Interview Questions & Answers

#### Q1: What is the purpose of `gr.Blocks()` in Gradio, and how does it compare to `gr.Interface()`?
**Answer:** `gr.Interface` is a simplified wrapper that provides a fixed side-by-side input/output layout. `gr.Blocks` is the underlying low-level layout API that allows developers to design custom, complex layouts using grids, rows, columns, tabs, accordion panels, and custom CSS styling.

#### Q2: How does a function using the `yield` keyword behave differently in Gradio than one using `return`?
**Answer:** A function with `return` exits immediately and can only send a single final output back to the UI. A function with `yield` acts as a generator. It can send multiple outputs sequentially over time without exiting. Gradio intercepts each `yield` value and updates the frontend UI elements in real-time, which is essential for displaying progress logs and streaming text.

#### Q3: Why is `scale` used inside `gr.Column()` blocks?
**Answer:** `scale` dictates the relative width of columns within a row. If Row has two columns with `scale=1` and `scale=2`, the second column will occupy two-thirds (66.6%) of the row's total width, while the first column occupies one-third (33.3%). This allows developers to build responsive, proportional layouts.

