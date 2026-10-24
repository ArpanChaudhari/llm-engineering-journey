# Day 16: LLM Leaderboards & Commercial Use Cases

## Overview

In this lesson, I learn how to compare LLMs before choosing one for an
application.

Main flow:

``` text
My use case
    ↓
Find candidate models
    ↓
Check leaderboards
    ↓
Compare quality, cost, and speed
    ↓
Test models on my own examples
    ↓
Choose the model that fits
```

## 🎯 Learning Objectives

-   Know which leaderboard to check and why.
-   Compare model intelligence, cost, and speed.
-   Understand human preference ratings in LM Arena.
-   Understand automation, augmentation, and agentic AI.

------------------------------------------------------------------------

# 1. Why Do We Need LLM Leaderboards?

There are many models. I need a quick way to shortlist them.

  Term          Meaning
  ------------- ---------------------------------------
  Benchmark     A test used to evaluate a model
  Leaderboard   A ranking/comparison of model results

**Remember:** A leaderboard helps me shortlist models. It does not make
the final decision for my application.

------------------------------------------------------------------------

# 2. Leaderboards Covered in This Lesson

## 2.1 Artificial Analysis

Website: https://artificialanalysis.ai/

**Use it when:** I want to compare model quality with price and speed.

Things to check: - Intelligence score - Input/output token pricing -
Time to first token - Output speed - Model version and evaluation date

### Example

*Fictional numbers for learning only.*

  Model       Quality on my test   Cost / 1,000 requests   Response time
  --------- -------------------- ----------------------- ---------------
  Model A                    95%                    ₹800           2.0 s
  Model B                    92%                    ₹180           0.8 s

If I need very high accuracy, A might be worth testing. If B already
meets my quality requirement, B may be the better choice because it is
cheaper and faster.

**Key point:** Highest intelligence score does not automatically mean
best business choice.

More details:
https://artificialanalysis.ai/methodology/intelligence-benchmarking

## 2.2 Hugging Face

Website: https://huggingface.co/

**Use it when:** I want to find and inspect models, especially
open-weight models.

Before using a model, check: - Model card and intended use - Model size
and hardware needs - Licence and commercial-use restrictions - Context
length and supported features - Evaluation results and update date

**Remember:** A model being available on Hugging Face does not
automatically mean it is unrestricted or easy to run locally.

## 2.3 Vellum

Website: https://www.vellum.ai/llm-leaderboard

**Use it when:** I want another model-comparison resource for
application development.

Before trusting a ranking, check: - Which models were compared? - What
tasks were tested? - When was it updated? - Does the test match my use
case?

## 2.4 SEAL

Website: https://scale.com/

SEAL is associated with Scale AI's model evaluation and leaderboard
work.

**What I should do:** 1. Open the specific SEAL evaluation. 2. Check
what capability it measures. 3. Check the date and model versions. 4.
Read how the score was produced.

**Remember:** A result from one evaluation is not a universal score for
every LLM task.

## 2.5 LiveBench

Website: https://livebench.ai/

**Main idea:** Evaluate models using challenging questions that are
refreshed periodically, helping reduce data-contamination risk.

Why this matters:

``` text
Old public question
       ↓
Model may have seen it during training
       ↓
Score may not show performance on new questions

Refreshed question
       ↓
Less chance of direct overlap
       ↓
More useful evidence of generalization
```

Check: - Overall score - Category scores, such as coding or
mathematics - Current release/date - Whether the category matches my
task

**Remember:** Fresh questions reduce contamination risk; they cannot
guarantee that contamination is impossible.

## 2.6 LM Arena / Arena

Website: https://arena.ai/leaderboard/text

**Main idea:** People compare model answers, often without knowing which
model produced each answer.

How it works:

``` text
User enters a prompt
       ↓
Model A answer  |  Model B answer
       ↓
User picks the preferred answer
       ↓
Many votes are collected
       ↓
Models receive relative ratings
```

Useful for: - Comparing conversational answers - Seeing which responses
people prefer - Finding models worth testing further

Limitation: - Preferred answer does not always mean factually correct
answer. - Results depend on prompts and voters.

------------------------------------------------------------------------

# 3. What Is Elo Rating?

Elo is a rating system originally used in chess. Arena-style rankings
use rating methods to estimate relative performance from comparisons.

Simple idea:

-   A model that wins comparisons tends to gain rating.
-   A model that loses comparisons tends to lose rating.
-   Many comparisons help estimate relative ranking.

**Do not confuse these:**

  -----------------------------------------------------------------------
  Elo-style rating                    Benchmark accuracy
  ----------------------------------- -----------------------------------
  Relative rating from comparisons    Score on a defined test

  Often based on human preference     Based on the benchmark's scoring
                                      rules

  Does not guarantee factual          Only measures the tasks tested
  correctness                         
  -----------------------------------------------------------------------

A small difference in ratings should not automatically be treated as
proof that one model is always better. Check the rating method and
uncertainty information where available.

Details: https://arena.ai/blog/ranking-method

------------------------------------------------------------------------

# 4. Quick Comparison: Which Site Should I Open?

  -----------------------------------------------------------------------
  My question                         Resource to check
  ----------------------------------- -----------------------------------
  Which model balances intelligence,  Artificial Analysis
  price, and speed?                   

  Where can I inspect or download     Hugging Face
  model weights?                      

  Where can I find another model      Vellum
  comparison?                         

  What does a particular SEAL         SEAL / Scale AI
  evaluation report?                  

  How do models perform on refreshed  LiveBench
  benchmark tasks?                    

  Which answer do people prefer?      Arena
  -----------------------------------------------------------------------

**My workflow:** Use leaderboards to shortlist candidates, then test
those candidates on my own examples.

------------------------------------------------------------------------

# 5. Commercial Use Cases

The lecture groups AI applications into three approaches.

## 5.1 Automation

AI completes a defined task with little human involvement.

**Example: Invoice processing**

``` text
Invoice PDF
    ↓
Extract invoice number, date, total
    ↓
Validate fields
    ↓
Save to database
    ↓
Send uncertain cases for review
```

Use when: - The task is repetitive. - Success criteria are clear. -
Results can be validated.

Risk: An incorrect value can enter the business system. Validation and
an error-handling path are important.

## 5.2 Augmentation

AI helps a person do work faster. The person stays involved.

**Example: Customer-support reply**

``` text
Customer question
       ↓
LLM drafts a reply
       ↓
Employee reviews/edits
       ↓
Employee sends reply
```

Use when: - AI can save time. - Human judgment is valuable. - The final
decision should remain with a person.

## 5.3 Agentic AI

An AI system can choose steps toward a goal and use tools to complete a
multi-step task within its permissions.

**Example: Investigate an order issue**

``` text
Customer reports a problem
       ↓
Agent looks up the order
       ↓
Agent checks the relevant policy
       ↓
Agent prepares a proposed resolution
       ↓
Human approval if required
       ↓
Action is completed
```

An agent needs more than good text generation. It also needs reliable
tool use, correct interpretation of tool results, error handling, and
clear permissions.

## 5.4 Difference at a Glance

  -----------------------------------------------------------------------
                    Automation        Augmentation      Agentic AI
  ----------------- ----------------- ----------------- -----------------
  Main role         Execute defined   Assist a person   Select steps
                    workflow                            toward a goal

  Human involvement Low for routine   Human             Depends on risk
                    cases             reviews/decides   and permissions

  Example           Extract invoice   Draft email for   Investigate order
                    data              review            with tools

  Main concern      Incorrect output  Human review      Tool/action
                                      quality           failures and
                                                        control
  -----------------------------------------------------------------------

These approaches can overlap in real applications.

------------------------------------------------------------------------

# 6. Match the Model to the Job

  Application          What I should test
  -------------------- ---------------------------------------------------
  Invoice extraction   Field accuracy, valid structure, cost
  Summarization        Correctness, important details retained
  Customer support     Helpful and correct replies, latency
  Coding assistant     Code correctness, tests passed
  RAG application      Retrieval quality, grounded answers
  Tool-using agent     Correct tool calls, task completion, safe actions

A general leaderboard score is not enough. My own evaluation should
match the work the model will do.

------------------------------------------------------------------------

# 7. Model Selection Checklist

Before choosing a model, I should check:

-   [ ] Does it perform well on my task?
-   [ ] Is the cost acceptable at my expected usage?
-   [ ] Is the response fast enough?
-   [ ] Does the licence allow my intended use?
-   [ ] Can I deploy it with my available hardware or API?
-   [ ] Does it fail on important edge cases?
-   [ ] Do I have a way to validate outputs and monitor production?

## Practical Workflow

``` text
Define task
   ↓
Set minimum quality requirement
   ↓
Use relevant leaderboards to shortlist
   ↓
Test candidates on the same examples
   ↓
Compare quality + cost + speed
   ↓
Choose and monitor
```

------------------------------------------------------------------------

# 8. Key Takeaways

1.  Benchmark = test; leaderboard = comparison of results.
2.  Artificial Analysis helps compare intelligence, price, and speed.
3.  Hugging Face helps me inspect models, licences, and model cards.
4.  LiveBench uses refreshed questions to reduce contamination risk.
5.  Arena measures human preference through answer comparisons.
6.  Elo-style ratings are relative ratings, not universal intelligence
    scores.
7.  Automation executes a defined workflow.
8.  Augmentation assists a human.
9.  Agentic AI can select steps and use tools toward a goal.
10. I must test models on my own use case before choosing one.

------------------------------------------------------------------------

# 9. Interview Questions

### Q1. What is the difference between a benchmark and a leaderboard?

A benchmark is a test. A leaderboard displays and compares model
results.

### Q2. Why use Artificial Analysis?

To compare model capability with practical factors such as price and
speed.

### Q3. What is LiveBench's main idea?

It uses refreshed evaluation questions to reduce the risk of models
having seen benchmark questions during training.

### Q4. What does Arena measure?

Human preferences between model answers, collected through comparisons.

### Q5. Does a higher Arena rating guarantee more factual answers?

No. People may prefer an answer that sounds better even when it is
incorrect.

### Q6. What is Elo?

A rating approach that estimates relative performance from wins, losses,
or ties in comparisons.

### Q7. What is the difference between automation and augmentation?

Automation performs a defined workflow; augmentation helps a person
perform a task.

### Q8. Why does agentic AI need extra safeguards?

It can call tools and take actions, so incorrect decisions can affect
external systems.

### Q9. How should I choose a model for a real application?

Use leaderboards to shortlist, then compare candidates on my own task
for quality, cost, speed, and reliability.

------------------------------------------------------------------------

# 10. Resources

-   Artificial Analysis: https://artificialanalysis.ai/
-   Hugging Face: https://huggingface.co/
-   Vellum LLM leaderboard: https://www.vellum.ai/llm-leaderboard
-   Scale AI: https://scale.com/
-   LiveBench: https://livebench.ai/
-   Arena text leaderboard: https://arena.ai/leaderboard/text
-   Arena ranking method: https://arena.ai/blog/ranking-method

Leaderboards and prices change. I should check the current date, model
version, and evaluation method before relying on a result.
