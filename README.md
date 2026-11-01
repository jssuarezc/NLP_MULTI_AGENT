## Repository Structure

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
│   ├── metrics.py
│   └── run_experiments.py   
└── results/
    ├── transcripts/
    ├── tables/
    └── plots/
```

### `data/`
All raw / processed data here 
* **`raw/`**: Full ARC-Challenge dataset (training, validation, and test splits)  -> inside .gitignore
* **`sampled/`**: The 100 development questions and 300 test questions used for the experiments are here. It also includes the manual categorization of factual, causal, comparison and numerical reasoning questions

### `notebooks/`
Jupyter notebooks for running the project specifications along with the outputs. Heavy logic is imported from `src/` to keep notebooks clean.
* **`01_dataset_exploration.ipynb`**: *STEP 1* - Used for initial dataset analysis (question length, option counts) and manual tagging.
* **`02_baseline_experiments.ipynb`**: *STEPS 2 & 3* - Runs the single-agent baselines and the independent multi-agent voting baseline.
* **`03_debate_experiments.ipynb`**: *STEPS 4 to 8* - Runs the debate protocol, LLM judge evaluation, tracking answer-change rates, and conducting the quality-cost analysis.

### `src/`
Has all python files + reusable helper modules
* **`config.py`**: Config for file paths, model names, LLM temperatures, and debate round limits.
* **`data_loader.py`**: Helper functions to download the ARC dataset and standardize the formatting of questions and options (A-D).
* **`prompts.py`**: File with system instructions and prompt templates (Standard Answering, Structured Criticism, LLM Judge).
* **`llm_client.py`**: API wrapper that handles LLM connectivity, enforces rate limits, counts tokens, and measures latency.
* **`debate_engine.py`**: Coordinates debate rounds, passing previous responses and critiques accurately between agents.
* **`metrics.py`**: Calculates majority voting, tie-breaking rules, and flip rates (correct $\leftrightarrow$ incorrect).
* **`run_experiments.py`**: Used to run the experiments without the need to run the notebook cells one after the other

### `results/`
All output is dumped here
* **`transcripts/`**: Full JSON/JSONL round-by-round conversation logs between the debating agents.
* **`plots/`**: Exported figures with top words, development and test content lengths, accuracy comparisons.
* **`tables/`**: All output tables from development and test outputs.


## SETTING UP
* Pull the repository and run the command below to create a local environment file for the project.

```bash
cp .env.example .env 
```
* Setup the model to use. Open the .env file and use either local model setup or enter openAI key.
* Run the command below to install the required package.

```bash
pip install -r requirements.txt
```

## RUNNING THE EXPERIMENT
There are two options. Through the notebook or Terminal + Notebook

### Notebook Option
* From the **notebooks** folder, open the notebooks one after the other starting from 01_dataset_exploration.ipynb to 02_baseline_experiments.ipynb to 03_debate_experiments.ipynb. Run each cell in 01 before moving to 02 before finally moving to 03.

### Terminal option
* Run the command 

```bash
python -m src.run_experiments --split dev --limit 10   # cheap end-to-end check to see if setup is all good
python -m src.run_experiments --split dev              # everything on the 100 dev questions
python -m src.run_experiments --split test             # final 300-question run
```

After the process is complete, proceed to the Notebook to run each of the cell to view the outputs. Click on run on each cell

**Note:** All tables, plots, transcripts are stored in the folder as you go.

**PS:** All outputs are generated using the following
* **Model:** mistral:7b
* **Dataset:** ARC-Challenge -> https://huggingface.co/datasets/allenai/ai2_arc