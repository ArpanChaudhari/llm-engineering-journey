# Day 16: LLM Model Selection, Scaling Laws & Benchmarks

## Overview

Today I am learning how to choose an LLM for a real application instead
of choosing a model only because it is popular or has a high benchmark
score.

The main idea is:

> The best model is the one that meets my application's quality
> requirements at an acceptable cost, speed, and level of risk.

I should consider model capability, pricing, latency, context length,
deployment options, privacy, and performance on my own task.

## Learning Objectives

-   Understand model parameters, training tokens, and inference.
-   Explain the main idea behind the Chinchilla scaling law.
-   Understand GPQA, MMLU-Pro, and Humanity's Last Exam (HLE).
-   Explain data contamination and overfitting.
-   Understand why benchmark scores are useful but not enough.
-   Build a simple model-evaluation plan for my own application.

## 1. What Does Choosing an LLM Mean?

Different models have different strengths. One might be good at coding,
another at following instructions, and another might offer a better
price or faster response.

Before choosing a model, I should answer:

1.  What task will the model perform?
2.  What level of accuracy do I need?
3.  How much latency can users tolerate?
4.  What is my expected input and output token usage?
5.  Does the data need to stay on my own machine or network?
6.  Does the model need tool calling, vision, a long context window, or
    structured output?
7.  How will I test whether it is good enough?

I should shortlist a few candidates and test them on the same
representative examples.

## 2. Parameters, Training Tokens, and Inference

### Parameters

Parameters are numerical values learned during model training. They are
part of the model's learned ability to recognize patterns and generate
text.

-   1B means approximately one billion parameters.
-   7B means approximately seven billion parameters.
-   70B means approximately seventy billion parameters.

A model with more parameters often has more capacity, but parameter
count alone does not determine quality. Training data, architecture,
training method, and post-training also matter.

### Training tokens

A token is a unit of text processed by a model. It may be a word, part
of a word, punctuation, or another text unit. The exact split depends on
the tokenizer.

Training tokens are the tokens used while training the model. The amount
and quality of training data affect what the model learns.

### Inference

Inference means using an already-trained model to produce an output from
an input prompt.

``` text
Training:
Data + Compute -> Learned Model

Inference:
Prompt -> Trained Model -> Generated Output
```

### Quick comparison

  Term               Meaning
  ------------------ ------------------------------------------------
  Parameters         Learned numerical values in the model
  Training tokens    Tokens processed during training
  Training compute   Computing resources used for training
  Inference          Running the trained model to get a result
  Latency            How long the model takes to respond
  Throughput         How much work the system can process over time

## 3. Chinchilla Scaling Law

### The question

Making a model bigger is not the only way to improve it. Researchers
studied how model size and training-data quantity should be balanced
when the training-compute budget is limited.

The Chinchilla research showed that some large language models were
undertrained relative to their parameter count. Under the compute budget
studied, a smaller model trained on more tokens could outperform a
larger model trained on fewer tokens.

The Chinchilla paper's 70B-parameter model was trained on about 1.4
trillion tokens and outperformed the 280B-parameter Gopher model on many
evaluated tasks.

### Rule of thumb

A commonly quoted approximation from the paper is:

``` text
Training tokens ≈ 20 × number of parameters
```

  Model size         Approximate tokens using this rule
  ---------------- ------------------------------------
  1B parameters                              20B tokens
  7B parameters                             140B tokens
  70B parameters                            1.4T tokens

This is a historical, approximate guideline for the training setup
studied. It is not a universal rule for every current model or every
training objective.

### What I should remember

-   Bigger does not automatically mean better.
-   Training data quantity and model size should be considered together.
-   Training compute is a constraint.
-   The best training balance depends on the model, data, objective, and
    compute budget.

## 4. Understanding AI Benchmarks

A benchmark is a standardized test used to evaluate some capability of a
model. A score only tells me how the model performed on that particular
test and setup.

### GPQA

GPQA is a difficult science question-answering benchmark with
graduate-level questions in areas such as biology, physics, and
chemistry. GPQA Diamond is a particularly challenging subset.

**What it helps investigate:** difficult scientific knowledge and
reasoning.

### MMLU-Pro

MMLU-Pro is a challenging multiple-choice benchmark covering many
academic and professional subjects. It was developed to make evaluation
more demanding than the original MMLU.

**What it helps investigate:** broad knowledge and reasoning across
subjects.

### Humanity's Last Exam (HLE)

HLE is a very challenging benchmark designed to probe the limits of
current AI systems across a broad range of expert-level subjects.

**What it helps investigate:** difficult questions that remain
challenging for advanced models.

### Important limitation

I should not compare scores from different benchmarks as if they were
the same test. Their questions, scoring methods, difficulty, and
coverage differ.

A model that scores higher on one benchmark is not automatically the
best model for my application.

## 5. Data Contamination and Overfitting

### Data contamination

Data contamination can happen when evaluation questions or their answers
overlap with the data used to train a model. The model may perform well
partly because it has already encountered the test material.

**Example:** A student sees the exact questions and answers before an
exam. A high score may not prove that the student can solve new
questions.

### Overfitting

Overfitting happens when a model adapts too closely to specific training
examples or patterns and does not generalize well to new examples.

**Example:** A model performs well on familiar coding questions but
fails when the same concept is presented in a different way.

### Difference

  -----------------------------------------------------------------------
  Data contamination                  Overfitting
  ----------------------------------- -----------------------------------
  Evaluation content overlaps with    The model relies too heavily on
  training data                       familiar patterns or examples

  Can make benchmark results look     Can cause weak performance on new
  better than general ability         or changed examples

  A concern about data exposure and   A concern about generalization
  evaluation validity                 
  -----------------------------------------------------------------------

These are related evaluation risks, but they are not the same thing.

## 6. How I Should Evaluate a Model

A more reliable evaluation process:

1.  Define the task and success criteria.
2.  Select several candidate models.
3.  Prepare representative examples from my intended use case.
4.  Run the same prompts and conditions for each candidate.
5.  Measure correctness and output quality.
6.  Measure latency and actual cost.
7.  Check failures, consistency, and safety.
8.  Test on new examples that were not used to tune the system.
9.  Choose the model that meets the requirements.
10. Monitor performance after deployment.

For example, if I build an invoice extraction system, I should measure
whether the model correctly extracts invoice number, date, tax, and
total. A general knowledge benchmark is not enough.

## 7. A Simple Model-Selection Example

These values are made up for learning. They are not real model benchmark
results or prices.

``` python
models = [
    {
        "name": "Model A",
        "accuracy": 95,
        "cost_per_1000_requests": 10,
        "latency_ms": 800,
    },
    {
        "name": "Model B",
        "accuracy": 92,
        "cost_per_1000_requests": 3,
        "latency_ms": 400,
    },
    {
        "name": "Model C",
        "accuracy": 85,
        "cost_per_1000_requests": 1,
        "latency_ms": 250,
    },
]

minimum_accuracy = 90

eligible_models = [
    model for model in models
    if model["accuracy"] >= minimum_accuracy
]

best_model = min(
    eligible_models,
    key=lambda model: model["cost_per_1000_requests"]
)

print(best_model)
```

Here Model B is selected because it meets the accuracy requirement and
costs less than Model A. This example uses a simple rule; a real
evaluation should also consider variability, privacy, task-specific
failure modes, and other requirements.

## 8. Key Takeaways

-   Parameter count alone does not tell me which model is best.
-   Training tokens and parameters are different concepts.
-   Chinchilla is about balancing model size and training data under a
    compute budget.
-   GPQA, MMLU-Pro, and HLE evaluate different kinds of challenging
    questions.
-   Benchmark scores depend on the benchmark and evaluation setup.
-   Data contamination and overfitting can make results misleading.
-   My own task-specific evaluation is essential.
-   I should compare quality, cost, speed, reliability, and deployment
    requirements together.

## 9. Interview Questions

### Q1. Do more parameters always mean a better model?

No. Parameter count is only one factor. Data quality, training,
architecture, and task alignment also affect performance.

### Q2. What is the Chinchilla scaling law about?

It studies how model size and training-data quantity should be balanced
under a fixed training-compute budget.

### Q3. What is the difference between parameters and tokens?

Parameters are learned numerical values in the model. Tokens are units
of text processed by the model.

### Q4. What is a benchmark?

A standardized evaluation used to measure performance on a particular
task or set of tasks.

### Q5. What is data contamination?

When evaluation material overlaps with training material, potentially
making benchmark performance look better than generalization.

### Q6. What is overfitting?

When a model fits familiar examples or patterns too closely and performs
poorly on new examples.

### Q7. Why test a model on my own data?

Because public benchmarks may not represent the exact tasks, data,
constraints, or failure costs of my application.

### Q8. How do I choose between a more capable model and a cheaper model?

I should first set a minimum quality requirement, then compare cost,
latency, reliability, and other constraints among models that meet it.

## 10. Further Reading

-   Chinchilla paper: https://arxiv.org/abs/2203.15556
-   GPQA paper: https://arxiv.org/abs/2311.12022
-   MMLU-Pro paper: https://arxiv.org/abs/2406.01574
-   Humanity's Last Exam: https://lastexam.ai/
-   LiveBench: https://livebench.ai/

I should use these sources to understand the methodology, not just
memorize the scores.
