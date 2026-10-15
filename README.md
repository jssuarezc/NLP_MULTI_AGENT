## Repository Structure

This is a structure proposal, feel free to tweak it too

```text
NLP_MULTI_AGENT/
├── data/
│   ├── raw/
│   └── sampled/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_baseline_experiments.ipynb
│   └── 03_debate_experiments.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── prompts.py
│   ├── llm_client.py
│   ├── debate_engine.py
│   └── metrics.py   
└── results/
    ├── transcripts/
    └── plots/
```

### `data/`
All raw / processed data here 
* **`raw/`**: Full ARC-Challenge dataset (training, validation, and test splits)  -> inside .gitignore
* **`sampled/`**: We could put the 100 development questions and 300 test questions used for our active experiments here.

### `notebooks/`
Jupyter notebooks for running the steps in the PDF. Heavy logic is imported from `src/` to keep notebooks clean.
* **`01_dataset_exploration.ipynb`**: *STEP 1* - Used for initial dataset analysis (question length, option counts) and manual tagging.
* **`02_baseline_experiments.ipynb`**: *STEPS 2 & 3* - Runs the single-agent baselines and the independent multi-agent voting baseline.
* **`03_debate_experiments.ipynb`**: *STEPS 4 to 8* - Runs the debate protocol, LLM judge evaluation, tracking answer-change rates, and conducting the quality-cost analysis.

### `src/`
Has all python files + reusable helper modules
* **`config.py`**: Config for file paths, model names, LLM temperatures, and debate round limits. -> working on it
* **`data_loader.py`**: Helper functions to download the ARC dataset and standardize the formatting of questions and options (A-D).
* **`prompts.py`**: File with system instructions and prompt templates (Standard Answering, Structured Criticism, LLM Judge).
* **`llm_client.py`**: API wrapper that handles LLM connectivity, enforces rate limits, counts tokens, and measures latency.
* **`debate_engine.py`**: Coordinates debate rounds, passing previous responses and critiques accurately between agents.
* **`metrics.py`**: Calculates majority voting, tie-breaking rules, and flip rates (correct $\leftrightarrow$ incorrect).

### `results/`
All output is dumped here
* **`transcripts/`**: Full JSON/JSONL round-by-round conversation logs between the debating agents. -> inside .gitignore
* **`plots/`**: Exported figures with accuracy comparisons, disagreement metrics, and cost-quality curves.

## KEEP IN MIND !!

Run this command once inside NLP_MULTI_AGENT so we avoid merging issues due to notebook options

```text
nbstripout --install
```