"""Central configuration: paths, models, generation settings, debate limits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(Path(__file__).resolve().parents[1])


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); never overrides variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

def _env(name: str, default: str) -> str:
    """Env value with inline `# comments` and whitespace removed; blank values fall back to the default."""
    return os.environ.get(name, "").split(" #")[0].strip() or default

# Directories
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLED_DIR = DATA_DIR / "sampled"
RESULTS_DIR = ROOT / "results"
TRANSCRIPT_DIR = RESULTS_DIR / "transcripts"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
CACHE_PATH = RESULTS_DIR / "cache" / "llm_cache.jsonl"

# Dataset
HF_DATASET = "allenai/ai2_arc"
HF_CONFIG = "ARC-Challenge"
SEED = 42
N_DEV = 100            # sampled from the official *validation* split
N_TEST = 300           # sampled from the official *test* split
DEV_SOURCE_SPLIT = "validation"
TEST_SOURCE_SPLIT = "test"

# Models / backend
MODEL = _env("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")
JUDGE_MODEL = _env("JUDGE_MODEL", MODEL)
BASE_URL = _env("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = _env("LLM_API_KEY", "ollama")
NUM_CTX = int(_env("LLM_NUM_CTX", "0"))   # 0 = do not send (OpenAI rejects unknown options)

# Client behaviour
MAX_WORKERS = int(_env("LLM_WORKERS", "2"))  # parallel questions / agent calls
MAX_RETRIES = 6


# Generation settings per call type
@dataclass(frozen=True)
class GenSpec:
    temperature: float
    max_tokens: int


# Single-agent baselines run at temperature 0 (deterministic reference).
# Anything that is *supposed* to differ between agents/samples uses T > 0,
# because with the same model and T = 0 "independent" agents would be clones.
GEN = {
    "direct": GenSpec(0.0, 32),
    "justify": GenSpec(0.0, 350),
    "sample_direct": GenSpec(0.7, 32),
    "sample_justify": GenSpec(0.7, 350),
    "debate": GenSpec(0.7, 350),
    "self_revise": GenSpec(0.7, 350),
    "judge": GenSpec(0.0, 350),
}

WORD_LIMIT = 100


# Debate
@dataclass(frozen=True)
class DebateConfig:
    n_agents: int = 3
    max_rounds: int = 2          # discussion rounds after the independent round 0
    protocol: str = "structured"  # "free" | "structured" by default
    early_stop: bool = False     # stop once all agents agree (optional extension)
    judge_final: bool = True     # judge sees anonymised final-round arguments
    judge_round0: bool = True    # judge on independent round-0 arguments (control)

N_AGENTS = 3
MAX_ROUNDS = 2
MAX_INDEP_SAMPLES = int(_env("MAX_INDEP_SAMPLES", "9"))   # extra independent samples for budget-matched baselines
TIE_BREAK = "hash"       # "hash" | "first_agent" | "lowest" 


def ensure_dirs() -> None:
    for p in (RAW_DIR, SAMPLED_DIR, TRANSCRIPT_DIR, PLOTS_DIR, TABLES_DIR, CACHE_PATH.parent):
        p.mkdir(parents=True, exist_ok=True)
