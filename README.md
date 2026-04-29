# TriCheck AI

TriCheck AI is a local multi-model LLM verification and benchmarking system built with Ollama.

The main goal of this project is to test whether multiple local LLMs can be combined to produce better final answers than a single model or simple majority voting.

Instead of relying on one model, the system asks three different local models the same benchmark question, compares their answers, and applies different decision strategies to select the final answer.

## Project Goal

Large language models can make mistakes when used alone. Different models may also fail on different questions. The goal of this project was to use the strengths of multiple local LLMs together and test whether a better final decision can be produced.

The project was designed to answer these questions:

- Can three local LLMs perform better together than a single model?
- Is simple majority voting enough?
- Can judge-based, review-based, or router-based decision systems improve accuracy?
- Which decision strategy works best on the same benchmark questions?

The final result showed that a carefully designed decision strategy can outperform both the strongest single model and naive majority voting.

## Models Used

The benchmark was run locally with Ollama using these models:

- `gemma3:12b`
- `llama3.1:8b`
- `mistral-nemo:12b`

These models were selected to test a local multi-model ensemble setup. Gemma was the strongest single model in the benchmark, but the best final result came from combining model outputs with a pattern-based routing strategy.

## Dataset: MMLU

This project uses a 100-question subset from the MMLU benchmark.

MMLU, which stands for Massive Multitask Language Understanding, is commonly used to evaluate language models across different subjects such as law, economics, mathematics, science, ethics, and general knowledge.

Using MMLU made the benchmark more realistic because the questions are not limited to one topic. The system had to handle different types of reasoning and knowledge-based questions.

The MMLU question file was generated with:

```bash
python build_mmlu_100.py
```

This creates:

```text
questions_mmlu_100.jsonl
```

For the final benchmark, the active question file is:

```text
questions.jsonl
```

## Evaluated Decision Systems

The project compares several decision-making systems. Each system receives the same benchmark questions and tries to produce the correct final answer.

### Single Model Baselines

Each model answers the question alone:

- `gemma3:12b`
- `llama3.1:8b`
- `mistral-nemo:12b`

This gives the baseline performance of each individual model.

### Majority Vote

All three models answer the same question. The answer selected by at least two models becomes the final answer.

This is a simple ensemble method, but it can fail when two weaker models agree on the wrong answer and override the stronger model.

### Confidence Weighted Vote

Each model gives an answer with a confidence score. The system tries to choose the answer with stronger confidence support.

This method tests whether model confidence can improve decision quality. However, the benchmark showed that confidence is not always reliable because models can give high confidence to wrong answers.

### Judge on Disagreement

If all models agree, the system accepts the shared answer. If the models disagree, a judge model is used to choose the final answer.

This method avoids extra judging when the models already agree, but still uses a judge when disagreement appears.

### Judge Always

A judge model reviews the answers for every question and selects the final answer.

This was one of the strongest systems. It performed better than the single-model baselines because it evaluated multiple model outputs instead of only trusting one answer.

### Reasoning Judge

The judge considers both the selected options and the reasoning behind the model answers.

The goal is to make a better final decision by looking at explanation quality, not only the final letter answer.

### Peer Review Revision

Each model first gives its own answer. Then the models see the other models' answers and revise their response.

This simulates a peer review process between models.

### Debate Revision

The models compare their answers and revise them after seeing the other responses.

This tests whether a debate-like process between models can improve the final answer.

### Gemma Final Arbiter

All model answers are passed to Gemma. Gemma acts as the final decision-maker and selects the final answer.

This uses the strongest single model as an arbiter over the other model outputs.

### Gemma Final Arbiter v2

This is a modified version of the Gemma arbiter prompt.

The result was lower than the original Gemma Final Arbiter, which suggests that making a prompt more detailed or stricter does not always improve accuracy.

### Judge Always v2

This is a modified version of the Judge Always system with a different judging prompt.

It performed worse than the original Judge Always system. This suggests that prompt changes can sometimes cause overcorrection or worse final decisions.

### Gemma Error-Check Revision

Gemma first gives an answer, then checks the other model outputs to see whether its answer should be revised.

This tests whether self-correction with external model feedback can improve performance.

### Peer Review + Gemma Final

The models first go through a peer review process. Then Gemma receives the reviewed outputs and makes the final decision.

This combines peer review with a strong final arbiter.

### Debate + Gemma Final

The models first go through a debate-style revision process. Then Gemma makes the final decision.

This tests whether debate outputs become more useful when passed to a final arbiter.

### Option-wise Verifier

The system evaluates answer options separately and tries to select the most supported option.

This approach was expected to help with reasoning, but in the final result it did not outperform the stronger judge-based methods.

### Adaptive Debate Stop

This system tries to use debate more selectively instead of applying the same process to every question.

The goal is to reduce unnecessary debate and only use extra reasoning when needed.

### Gemma Self-Consistency 3x

Gemma answers the same question multiple times. The system then selects the most consistent answer.

This performed strongly, but it required more time because the model is called multiple times.

### Gemma-Centered Sequential Review

This system uses Gemma as the central model.

The process is:

1. Gemma gives an initial answer.
2. Llama gives its own answer and reviews Gemma's answer.
3. Mistral gives its own answer and reviews the previous answers.
4. Gemma receives all responses again.
5. Gemma makes the final decision.

This tests whether a structured multi-model review process can improve accuracy.

### Coarse Pattern-Best Router

This was the best-performing system.

The router looks at the answer pattern between the three models, such as:

- All models agree
- Gemma and Llama agree
- Gemma and Mistral agree
- Llama and Mistral agree while Gemma disagrees
- All models give different answers

Based on the pattern, the router selects the decision strategy that worked best for that type of situation.

This method does not simply trust the majority. Instead, it uses the structure of model agreement and disagreement to choose a better final decision strategy.

## Final Result Summary

The benchmark was run on a 100-question MMLU subset.

| System | Accuracy | Avg Extra Time |
|---|---:|---:|
| Coarse Pattern-Best Router | 69% | 0.00s |
| Judge Always | 68% | 17.31s |
| Gemma Self-Consistency 3x | 68% | 35.20s |
| Judge on Disagreement | 67% | 11.51s |
| Gemma Final Arbiter | 66% | 11.24s |
| TriCheck Meta Decision | 65% | 0.00s |
| Peer Review + Gemma Final | 65% | 10.83s |
| Peer Review Revision | 64% | 20.05s |
| gemma3:12b | 63% | 3.84s |
| Reasoning Judge | 63% | 15.25s |
| Gemma-Centered Sequential Review | 62% | 33.59s |
| Adaptive Debate Stop | 61% | 1.05s |
| Gemma Final Arbiter v2 | 60% | 16.75s |
| Confidence Weighted Vote | 59% | 0.00s |
| Debate + Gemma Final | 59% | 11.71s |
| Gemma Error-Check Revision | 59% | 9.73s |
| Debate Revision | 59% | 28.07s |
| Majority Vote | 58% | 0.00s |
| Option-wise Verifier | 58% | 34.08s |
| mistral-nemo:12b | 58% | 7.47s |
| llama3.1:8b | 56% | 4.62s |
| Judge Always v2 | 54% | 9.79s |

## Key Findings

The strongest single model was `gemma3:12b` with 63% accuracy.

Simple majority voting reached only 58%, which was lower than Gemma alone. This shows that using more models does not automatically improve performance.

The best result came from the Coarse Pattern-Best Router with 69% accuracy.

This means the best system improved performance by:

- +6 points over Gemma alone
- +11 points over Majority Vote

The main finding is that local LLM ensemble performance depends more on the decision strategy than simply adding more models.

## Error Pattern Analysis

The benchmark results showed several important patterns.

### Majority voting was not always reliable

Majority Vote failed when two models agreed on the same wrong answer.

In the benchmark, there were questions where only one of the three base models selected the correct answer. In these cases, simple majority voting always failed because the two incorrect models outvoted the correct one.

This is one reason why Majority Vote stayed at 58%.

### Model confidence was not fully reliable

Confidence Weighted Vote only reached 59%.

The reason is that models often gave high confidence even when their answers were wrong. This means confidence scores alone were not enough to decide which model should be trusted.

This suggests that confidence-based voting can be risky when local LLMs are overconfident in incorrect answers.

### Judge-based systems worked better

Judge Always reached 68%, and Judge on Disagreement reached 67%.

These systems performed better because they did not only count votes. Instead, they reviewed the model outputs and selected the answer that seemed most correct.

This was especially useful when the models disagreed.

### More complex prompting did not always help

Some more complex systems performed worse than simpler judge-based systems.

For example:

- Judge Always: 68%
- Judge Always v2: 54%
- Gemma Final Arbiter: 66%
- Gemma Final Arbiter v2: 60%
- Debate Revision: 59%
- Option-wise Verifier: 58%

This suggests that longer or more complicated prompting can sometimes introduce overthinking, overcorrection, or unstable decisions.

The benchmark did not directly measure hallucination, but lower performance in some revision and debate systems suggests that extra reasoning steps can sometimes move the model away from the correct answer instead of improving it.

### Pattern-based routing was the best approach

The Coarse Pattern-Best Router achieved the highest score.

The reason is that it used the agreement pattern between models. It did not always trust the majority, and it did not always call the same judge strategy.

For example, when Gemma disagreed with Llama and Mistral, the router could avoid blindly trusting the two-model majority. This helped recover some cases where the majority answer was wrong.

The router was correct in several cases where Majority Vote failed, which explains why it reached the best overall score.

## Example Observations

Some examples from the result file show why routing is useful.

In some questions, Gemma selected the correct answer while Llama and Mistral agreed on the wrong answer. Majority Vote selected the wrong answer, but the router selected the correct one.

In other questions, all three models disagreed. In this case, Majority Vote could not produce a useful majority decision, while judge-based systems and the router were still able to choose a final answer.

These examples show that the important part is not only having multiple models, but knowing when to trust which decision strategy.

## Category and Difficulty Notes

All 100 questions in this benchmark subset were marked as medium-hard.

The category-level results should be interpreted carefully because some MMLU categories had only a small number of questions in the 100-question subset.

However, the benchmark still covered multiple subject areas such as law, ethics, economics, mathematics, science, and general knowledge. This made the test more useful than a single-topic benchmark.

## Project Structure

```text
local-llm-ensemble-benchmark/
│
├── benchmarks/
│   └── benchmark_full100_all_plus_v2_no_critical.py
│
├── results/
│   ├── benchmark_by_category_gemma_mmlu_full100_all_plus_v2_no_critical.csv
│   ├── benchmark_by_difficulty_gemma_mmlu_full100_all_plus_v2_no_critical.csv
│   ├── benchmark_results_gemma_mmlu_full100_all_plus_v2_no_critical.csv
│   └── benchmark_summary_gemma_mmlu_full100_all_plus_v2_no_critical.csv
│
├── build_mmlu_100.py
├── questions.jsonl
├── questions_mmlu_100.jsonl
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is installed and the required models are available:

```bash
ollama pull gemma3:12b
ollama pull llama3.1:8b
ollama pull mistral-nemo:12b
```

Generate the MMLU subset if needed:

```bash
python build_mmlu_100.py
```

Run the final benchmark:

```bash
python benchmarks/benchmark_full100_all_plus_v2_no_critical.py
```

## Technologies Used

- Python
- Ollama
- Local LLMs
- MMLU benchmark subset
- CSV result analysis
- Multi-model decision strategies

## Limitations

This benchmark uses a 100-question MMLU subset, so the results should be seen as an experimental comparison rather than a universal final result.

The results may change with:

- Different MMLU samples
- Different local models
- Different prompts
- Different Ollama versions
- Larger benchmark sizes

The project does not claim that one strategy is always best for every task. Instead, it shows that decision strategy design has a major effect on local LLM ensemble performance.

## Conclusion

TriCheck AI shows that a carefully designed local LLM ensemble can outperform both single-model baselines and simple voting.

The best-performing approach was not majority voting, but a pattern-based router that selects the most suitable decision strategy depending on how the models agree or disagree.

The final result supports the main idea of the project: using multiple local LLMs can improve performance, but only when their outputs are combined with a strong decision strategy.
