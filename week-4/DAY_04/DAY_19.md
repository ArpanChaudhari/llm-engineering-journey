# Day 19 — Python vs Java Performance Benchmark with Groq and Gradio

## 1. What we worked on

In this notebook, we explored how to use an LLM to translate a Python program into Java and then compare the execution speed of the two versions. We also built a small Gradio interface to request Java code from a model, save the generated file, and run a benchmark.

The main goal was not simply to generate code. We wanted to check whether the Java version produced the correct result and measure how long it took to run compared with Python.

## 2. Learning objectives

- Understand how an LLM API can be used for code translation.
- Compare two models for the same code-generation task.
- Understand the role of a prompt in guiding generated code.
- Use a simple Gradio interface to interact with a Python function.
- Compile and run Java from a Python workflow.
- Validate program output before trusting benchmark timings.
- Calculate speedup and summarize results in a leaderboard.
- Save benchmark results to a CSV file.

## 3. Overall workflow

1. Configure access to the Groq API.
2. Select the language models to test.
3. Prepare a Python benchmark program.
4. Ask each model to translate the Python program into Java.
5. Save the generated Java source file.
6. Compile the Java file using the Java compiler.
7. Run the Java program and check its output.
8. Measure the Python and Java execution times.
9. Repeat timed runs and use the median to reduce the effect of unusually slow or fast runs.
10. Calculate speedup, rank the results, and export the leaderboard.

## 4. API and model configuration

The notebook uses the Groq API to send prompts to language models. API access is configured through an API key, which should be kept private and should not be written directly into shared source files.

Two models are configured in the notebook:
- GPT-OSS 120B
- Qwen

The purpose of testing more than one model is to compare the Java code each model produces under the same task. A model that produces code quickly is not automatically the best: its code must compile, produce the expected result, and perform well.

## 5. The code-translation function

The notebook defines a function named `port()` that sends the Python program and instructions to a selected model. The model is asked to return a Java translation.

The prompt is important because it tells the model what needs to be preserved. For a fair comparison, the Java program should implement the same calculation as the Python program, use compatible input values, and produce an output that can be checked.

Generated code should always be reviewed. LLM output can contain syntax errors, use a different algorithm, or fail to preserve the original program's behavior.

## 6. Gradio user interface

Gradio is used to create a small interface around the translation workflow. It lets a user choose a model, request a translation, and work with the generated Java source without having to call the function manually.

This demonstrates how a Python function can be turned into a simple interactive tool. Gradio provides the interface; the model API performs the translation; and the benchmark functions handle compilation, execution, and timing.

## 7. The benchmark task

The notebook uses a numerical calculation that approximates pi with 200 million iterations. Python runs the reference version, and the generated Java program runs the corresponding calculation.

A benchmark needs a defined task so both implementations can be compared. If the two programs do different amounts of work, their timings do not represent a fair language comparison.

The expected result is checked before a timing is accepted. Correctness matters more than speed: a very fast program is not useful if it calculates the wrong answer.

## 8. Compiling and running Java

The notebook invokes `javac` to compile the generated Java source. Compilation errors must be handled before the program can be benchmarked. After compilation, the Java program is executed and its output is captured for checking.

This is an example of Python orchestrating an external tool. Python manages the experiment, while the Java compiler and Java runtime perform their respective jobs.

## 9. Correctness validation

The benchmark compares program output against the expected result or reference output. This helps catch code-generation mistakes before they are mistaken for performance improvements.

Output checks should account for the fact that numerical calculations may produce small floating-point differences. A suitable tolerance can be more appropriate than requiring every digit to match exactly.

A stronger benchmark would also verify that both programs use the same algorithm, iteration count, and numerical method.

## 10. Timing methodology

The notebook uses:
- **Warm-up runs:** 3
- **Timed repetitions:** 5
- **Summary statistic:** median execution time

Warm-up runs help reduce the influence of initial startup effects. Repeating measurements helps reveal timing variability. The median is less affected than the mean by a single unusually high or low measurement.

Execution time can still depend on the computer, background processes, runtime startup, compiler settings, and other environmental factors. Therefore, these results describe this particular experiment rather than proving that one language is always faster.

## 11. Understanding speedup

Speedup is calculated by dividing the Python execution time by the Java execution time.

A speedup greater than 1 means the Java version completed the measured task faster in this experiment. For example, a speedup of about 10× means the Python time was roughly ten times the Java time.

The speedup should only be interpreted after correctness has been established and the task is comparable. It does not mean that every Java program will be faster than every Python program.

## 12. Results recorded in the notebook

The notebook recorded the following approximate results:

| Implementation | Execution time | Speedup vs Python |
|---|---:|---:|
| Python baseline | 50.508291 seconds | 1× baseline |
| Qwen-generated Java | 0.981000 seconds | 51.49× |
| GPT-OSS 120B-generated Java | 1.427452 seconds | 35.38× |

In this run, the Qwen-generated Java version was the faster of the two generated Java versions. These figures are measurements from the notebook's run, not universal performance guarantees.

## 13. Leaderboard and CSV export

The notebook organizes the measured times and speedup values into a leaderboard so the implementations can be compared in one place. It also exports benchmark results to a CSV file.

CSV is useful because the results can be opened in a spreadsheet or loaded into another Python analysis workflow. A good results table should identify the model, implementation, timing, correctness status, and speedup so that the numbers are interpretable later.

## 14. Important limitation: reusing saved Java

The notebook has a setting named `reuse_saved_java=True`. When this is enabled, a previously saved Java file may be reused instead of generating a fresh translation.

This is convenient when rerunning a benchmark, but it creates a potential problem: the Java file might not correspond to the current Python source or the selected model. Before trusting a result, confirm which Java file was actually compiled and benchmarked. For a clean model comparison, record whether the Java file was newly generated or reused.

## 15. Limitations and possible improvements

- Test more than one benchmark task; one numerical loop cannot represent all workloads.
- Confirm that each model's Java implementation performs equivalent work.
- Keep the computer and runtime conditions as consistent as possible.
- Record compilation time separately from execution time.
- Distinguish a cold-start measurement from steady-state execution.
- Store the exact generated Java source with each result.
- Record model names, prompt versions, correctness checks, and timing repetitions.
- Avoid exposing API keys in code, logs, screenshots, or shared files.
- Run generated code only in an environment appropriate for code you have not fully reviewed.

## 16. Key takeaways

1. An LLM can help translate code, but generated output still needs review.
2. A successful API response does not guarantee valid or correct code.
3. Compilation confirms syntax and some type checks, but it does not prove the result is correct.
4. Correctness validation should happen before performance comparison.
5. Warm-ups and repeated measurements make timing more reliable.
6. Median time is useful when individual runs vary.
7. Speedup is meaningful only when both versions perform equivalent work.
8. Results from one benchmark should not be generalized to every program.
9. Reusing a saved generated file can affect whether the comparison is truly testing the selected model.
10. A leaderboard and CSV make the experiment easier to inspect and reproduce.

## 17. Interview questions and answers

### Q1. What was the purpose of this notebook?
**Answer:** To use LLMs to translate a Python benchmark into Java, validate the generated program, compare execution times, and summarize the results.

### Q2. Why did we use Groq?
**Answer:** The notebook uses Groq as the API service through which prompts are sent to the selected language models and their responses are received.

### Q3. Why test two models?
**Answer:** To compare the Java code produced by different models for the same task. Their outputs can differ in correctness, implementation choices, and execution performance.

### Q4. What is the role of the prompt?
**Answer:** It describes the translation task and the behavior the generated Java code should preserve. Clear instructions help the model produce code that is easier to validate.

### Q5. Why is Gradio used?
**Answer:** It provides a simple interactive interface for using the translation function without requiring the user to call the Python function directly.

### Q6. What does `javac` do?
**Answer:** It compiles Java source code into a form that the Java runtime can execute. If compilation fails, the program should not proceed to the timing stage.

### Q7. Why check correctness before measuring speed?
**Answer:** A program that returns the wrong answer cannot be considered a valid optimization. Correctness must be established before timing results can be trusted.

### Q8. Why use warm-up runs?
**Answer:** Initial runs can be affected by startup and runtime effects. Warm-ups help reduce the influence of these effects on the measured repetitions.

### Q9. Why repeat the benchmark five times?
**Answer:** Individual runs can vary. Repeated measurements provide a more useful picture of typical performance than relying on a single run.

### Q10. Why use the median instead of only one timing?
**Answer:** The median is less sensitive to an unusually slow or fast run and gives a robust summary of the repeated measurements.

### Q11. How is speedup calculated?
**Answer:** Divide the Python execution time by the Java execution time. A larger value means the Java version was faster for the measured task.

### Q12. Does this prove Java is always faster than Python?
**Answer:** No. The notebook measures one particular workload in one environment. Results depend on the algorithm, implementation, runtime, and hardware.

### Q13. What is a fair benchmark?
**Answer:** Both implementations should perform equivalent work, use the same inputs and comparable algorithms, and be measured under consistent conditions.

### Q14. Why might generated code compile but still be wrong?
**Answer:** Compilation checks language syntax and certain type rules, but it cannot guarantee that the program follows the intended logic or produces the expected answer.

### Q15. What is the risk of reusing a saved Java file?
**Answer:** The file might have been generated from an older version of the Python program or by a different model, so the benchmark might not be testing what we think it is.

### Q16. What does the recorded result show?
**Answer:** In this notebook run, Python took about 50.51 seconds, Qwen-generated Java took about 0.981 seconds, and GPT-OSS 120B-generated Java took about 1.427 seconds. Qwen's Java version was faster in this particular experiment.

### Q17. Why export results to CSV?
**Answer:** CSV makes the measurements easy to inspect in spreadsheet software and reuse for further analysis.

### Q18. What improvements would make the benchmark stronger?
**Answer:** Use multiple tasks, confirm equivalent algorithms, save the exact source code, record environment details, separate compilation from execution time, and report timing variation.

### Q19. What should be protected when using an API?
**Answer:** The API key should be kept secret and should not be committed to a public repository or included in shared files.

### Q20. What is the most important lesson from this notebook?
**Answer:** First verify that generated code is correct; only then compare performance using repeatable measurements.
