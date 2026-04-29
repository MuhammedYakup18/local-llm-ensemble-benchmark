# TriCheck AI

TriCheck AI is a local multi-model LLM verification and benchmarking system built with Ollama.

The main goal of this project is to test whether using multiple local LLMs together can produce better answers than relying on a single model or simple majority voting.

Instead of only asking one model, the system asks three different local models the same MMLU question, compares their answers, and then applies different decision strategies to choose the final answer.

## Project Goal

Large language models can make mistakes, especially when they are used alone. The idea behind this project is that different models may have different strengths and weaknesses.

Our goal was to:

- Use three local LLMs together
- Compare different ensemble decision strategies
- Test whether model collaboration can improve accuracy
- Find a better method than naive majority voting
- Benchmark all systems on the same 100-question MMLU subset

The final result showed that the best strategy achieved higher accuracy than both the strongest single model and simple majority voting.

## Models Used

The benchmark was run locally with Ollama using these models:

- `gemma3:12b`
- `llama3.1:8b`
- `mistral-nemo:12b`

Gemma was the strongest single model in the benchmark, but the best overall result came from combining model outputs with a pattern-based routing strategy.

## Dataset

The project uses a 100-question subset from the MMLU benchmark.

The MMLU questions were generated with:

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

The project compares multiple decision-making strategies. These systems were designed to test whether three local models can work together and produce a better final answer.

### Single Model Baselines

Each model answers the question alone.

- `gemma3:12b`
- `llama3.1:8b`
- `mistral-nemo:12b`

This shows the base performance of each individual model.

### Majority Vote

All three models answer the same question. The answer chosen by at least two models becomes the final answer.

This is a simple ensemble method, but it can fail when two weaker models agree on the wrong answer.

### Confidence Weighted Vote

Each model gives an answer with a confidence score. The system tries to select the answer with stronger confidence support.

This method tests whether model confidence can help improve final decision quality.

### Judge on Disagreement

If all models agree, the system accepts the shared answer. If the models disagree, a judge model is used to choose the final answer.

This avoids unnecessary judging when the models already agree.

### Judge Always

A judge model reviews the answers for every question and selects the final answer.

This method performed strongly because the judge evaluates all model outputs instead of only using voting.

### Reasoning Judge

The judge considers not only the final options selected by the models, but also their reasoning.

The goal is to make a more informed final decision based on explanation quality.

### Peer Review Revision

Each model first gives its own answer. Then the models see the other models’ answers and revise their response.

This simulates a peer review process between models.

### Debate Revision

The models compare and revise their answers after seeing the other responses.

This method tests whether debate-like interaction can improve the final answer.

### Gemma Final Arbiter

All model answers are passed to Gemma. Gemma acts as the final decision-maker and selects the final answer.

This uses the strongest single model as an arbiter over the other models.

### Gemma Self-Consistency 3x

Gemma answers the same question multiple times. The system then chooses the most consistent answer.

This tests whether repeated reasoning from the strongest model can improve reliability.

### Gemma-Centered Sequential Review

This system uses Gemma as the central model.

The process is:

1. Gemma gives an initial answer.
2. Llama gives its own answer and reviews Gemma’s answer.
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
- Gemma disagrees with both models
- All models give different answers

Based on the pattern, the router chooses the decision strategy that worked best for that type of situation.

This method does not simply trust the majority. Instead, it uses the structure of model agreement and disagreement to choose a better final decision strategy.

## Final Result Summary

The benchmark was run on a 100-question MMLU subset.

| System | Accuracy |
|---|---:|
| Coarse Pattern-Best Router | 69% |
| Judge Always | 68% |
| Gemma Self-Consistency 3x | 68% |
| Judge on Disagreement | 67% |
| Gemma Final Arbiter | 66% |
| Gemma3:12B | 63% |
| Majority Vote | 58% |
| Mistral-Nemo:12B | 58% |
| Llama3.1:8B | 56% |

## Key Finding

The main finding is that using multiple models is not automatically better.

Simple majority voting reached only 58%, which was weaker than Gemma alone at 63%.

The best result came from the Coarse Pattern-Best Router, which reached 69%.

This means the final router improved performance by:

- +6 points over Gemma3:12B alone
- +11 points over Majority Vote

The result suggests that local LLM ensemble performance depends more on choosing the right decision strategy than simply adding more models.

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

## Conclusion

TriCheck AI shows that a carefully designed local LLM ensemble can outperform both single-model baselines and simple voting.

The best-performing approach was not majority voting, but a pattern-based router that selects the most suitable decision strategy depending on how the models agree or disagree.
