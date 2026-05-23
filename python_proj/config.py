import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Models - All use Godfather now
PRO_MODEL_1 = "godfather/llama3:latest"
PRO_MODEL_2 = "godfather/llama3:latest"
CON_MODEL_1 = "dolphin-mistral:7b"
CON_MODEL_2 = "dolphin-mistral:7b"
JUDGE_MODEL = "godfather/gpt-oss:20b"

# Godfather API
GODFATHER_ENDPOINT = os.getenv("GODFATHER_ENDPOINT", "http://164.52.211.24/yashas/models")
GODFATHER_DELAY_SECONDS = float(os.getenv("GODFATHER_DELAY_SECONDS", "1.0"))

# Web Search
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "duckduckgo")
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "8"))
MAX_SCRAPED_PAGES_PER_FACTOR = int(os.getenv("MAX_SCRAPED_PAGES_PER_FACTOR", "5"))
SCRAPE_TIMEOUT = int(os.getenv("SCRAPE_TIMEOUT", "15"))

# Debate
DEBATE_ROUNDS = int(os.getenv("DEBATE_ROUNDS", "3"))
MAX_ARGUMENT_LENGTH = int(os.getenv("MAX_ARGUMENT_LENGTH", "200"))
ALLOW_CROSS_CRITIQUE = os.getenv("ALLOW_CROSS_CRITIQUE", "true").lower() == "true"
MAX_FACTORS = int(os.getenv("MAX_FACTORS", "5"))

# Rate Limiting
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "0.5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF_FACTOR = float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0"))

# Evaluation
ENABLE_ANONYMIZATION = os.getenv("ENABLE_ANONYMIZATION", "true").lower() == "true"
SCORING_SCALE = os.getenv("SCORING_SCALE", "1-10")

# Output
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/")
SAVE_TRANSCRIPTS = os.getenv("SAVE_TRANSCRIPTS", "true").lower() == "true"
