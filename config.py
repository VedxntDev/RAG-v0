import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CORPUS_PATH = DATA_DIR / "sample_corpus.json"
TEST_UTTERANCES_PATH = DATA_DIR / "test_utterances.json"

# Search & Retrieval Parameters
BM25_TOP_K = 10
DENSE_TOP_K = 10
RRF_K = 60
FUSED_TOP_K = 5

# Controller Thresholds
MIN_STABILITY_WORDS = 3
PRESENTATION_KEYWORDS = [
    "reformat", "bullet", "bullets", "summarize", "shorten",
    "translate", "rewrite", "simplify", "format as", "in spanish", "in french"
]
INCOMPLETE_TRAILING_WORDS = [
    "in", "to", "for", "with", "at", "on", "and", "or", "a", "an", "the", "if", "is", "are"
]

# LLM & Cost Estimation Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # 'auto', 'gemini', 'openai', 'mock'
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")

COST_PER_1K_INPUT_TOKENS = 0.00015
COST_PER_1K_OUTPUT_TOKENS = 0.0006
