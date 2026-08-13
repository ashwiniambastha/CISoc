"""
Central configuration for the cross-lingual ontology probe.

Everything that could change the behaviour of a call lives here, so that the
run manifest can record it in one place (Appendix C.7.6).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Provider -------------------------------------------------------------
BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY_ENV = "NVIDIA_API_KEY"


def get_api_key() -> str:
    key = os.getenv(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} not found. Put it in {PROJECT_ROOT / '.env'} as "
            f"{API_KEY_ENV}=nvapi-..."
        )
    return key


# --- Models ---------------------------------------------------------------
# rpm_limit: confirm against your own build.nvidia.com dashboard before a full
# run. NVIDIA describes the free tier as traffic-dependent, not a hard SLA.
MODELS = {
    "nemotron": {
        # Catalog name confirmed via list_models.py on 2026-08-13. The spec's
        # "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" 404s: the API
        # uses the lowercase form without the quantisation suffix.
        "model_id": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "rpm_limit": 40,
    },
    "gpt_oss": {
        "model_id": "openai/gpt-oss-120b",
        "rpm_limit": 30,
    },
}

# --- Elicitation protocol (spec section 4) --------------------------------
# DEFAULT_REPLICATES is the n that Appendix B.1 says the pilot exists to
# calibrate. It is the only thing that gives you within-prompt stability p-hat,
# and the 70% recurrence rule is computed against it. Lowering it to 3 makes
# the threshold 2/3 = 67% or 3/3 = 100%, with nothing in between.
DEFAULT_REPLICATES = 8          # temperature left unset
DETERMINISM_REPLICATES = 1      # temperature = 0
DETERMINISM_TEMPERATURE = 0.0
DETERMINISM_TOP_P = 1.0
DETERMINISM_SEED = 42           # honoured only if the endpoint supports seeding

# --- Temperature sweep (exploratory, NOT the protocol run) ----------------
# A separate condition, logged as condition="sweep" with the actual temperature
# on every row. Sweep rows are not pooled with `default` rows: the default
# condition is defined by sending no temperature at all, which is not the same
# event as sending a number that happens to match the provider's default.
SWEEP_TEMPERATURES = [0.0, 0.7, 1.4]
SWEEP_REPLICATES = 1            # replicates per temperature

# Identical across both models, recorded on every row (Appendix C.7.3).
MAX_TOKENS = 4096

# --- Response-length frame -------------------------------------------------
# THIS IS PRIMING, AND IT IS ON THE RECORD AS SUCH.
#
# C.3 excludes framing sentences. B.3 permits an added frame only on condition
# that it "must be logged and its effect on the assertion rate reported". So the
# frame is versioned, the exact string sent is written into every row, and the
# manifest records it.
#
# Known risk, to be reported rather than assumed away: brevity pressure
# compresses attribution before it compresses assertion — "some traditions hold
# that..." is the first clause a model drops when told to be short — so the
# frame plausibly inflates the measured assertion rate. Rows carrying different
# frame_id values are never pooled.
#
# Set USE_BREVITY_FRAME = False to return to the unframed protocol.
USE_BREVITY_FRAME = True
BREVITY_FRAME_ID = "brief-v1"
BREVITY_FRAME_PLACEMENT = "suffix"     # always after the prompt, never before

# Same-language by construction. An English instruction attached to a Hindi
# prompt would confound the language comparison with the language of the frame.
BREVITY_FRAMES = {
    "EN": "Answer in two or three sentences.",
    "HI": "दो या तीन वाक्यों में उत्तर दीजिए।",
    "HG": "Do ya teen vaakyon mein jawab dijiye.",
}
BREVITY_FRAME_SEPARATOR = "\n\n"

# Transport-failure policy: one retry, then log failed and move on (spec 4.5).
MAX_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 5.0
REQUEST_TIMEOUT_SECONDS = 180.0

# --- Rate limiting (spec section 5) ---------------------------------------
CALLS_BEFORE_PAUSE = 20
PAUSE_SECONDS = 15

# --- Ordering (Appendix C.7.2) --------------------------------------------
# Prompt order is re-randomised per replicate; the seed is derived from this
# base and written into every row so the order is reproducible.
ORDER_SEED_BASE = 20260813

# --- Paths ----------------------------------------------------------------
PROMPT_BANK_PATH = PROJECT_ROOT / "prompt_bank.json"
LOG_DIR = PROJECT_ROOT / "logs"
EXCEL_LOG_PATH = LOG_DIR / "run_log.xlsx"
JSONL_LOG_PATH = LOG_DIR / "run_log.jsonl"
MANIFEST_PATH = LOG_DIR / "run_manifest.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Log schema (spec section 6, plus Appendix C hygiene fields) ----------
# RECORD_FIELDS is what every log record contains and what goes to the JSONL.
# EXCEL_COLUMNS (below) is the narrower view written to the spreadsheet. They
# are deliberately separate: hiding a column from the sheet should never delete
# it from the record.
#
# Ordered for reading, not for machine convenience: what a coder needs to see
# without scrolling comes first — which prompt, in which language, to which
# model, what was sent, what came back. Housekeeping trails at the right.
RECORD_FIELDS = [
    # --- what was asked -----------------------------------------------
    "prompt_id",
    "language",            # EN | HI | HG
    "arm",                 # test | control
    "prompt_text",         # the bank prompt, unmodified
    "prompt_sent",         # what actually went over the wire, frame included
    # --- who was asked, and how ---------------------------------------
    "model_name",          # nemotron | gpt_oss
    "condition",           # default | determinism | sweep | manual
    "temperature",         # value sent, or "unset"
    "replicate_index",
    # --- what came back ------------------------------------------------
    "response_text",       # verbatim response, or error message if failed
    "status",              # success | failed
    "truncated",
    "response_chars",
    "total_tokens",
    # --- context -------------------------------------------------------
    "family",
    "form",                # contrastive | classification | boundary
    "paraphrase",          # a | b
    "frame_id",            # none | brief-v1 ... never pool across values
    "frame_text",          # the exact instruction appended, verbatim
    "timestamp",
    "source",              # batch | dashboard
    # --- housekeeping ---------------------------------------------------
    "prompt_tokens",
    "completion_tokens",
    "latency_seconds",
    "finish_reason",
    "attempt",             # 1 = first try, 2 = the single retry
    "top_p",
    "max_tokens",
    "model_id_sent",
    "model_id_returned",
    "order_seed",
    "bank_version",
    "bank_sha256",
    "record_id",
]

# Columns actually written to run_log.xlsx. Anything listed here is a view of
# RECORD_FIELDS; anything left out is still captured in run_log.jsonl.
#
# prompt_sent is hidden because it duplicates prompt_text once the frame is
# known — and the frame is recoverable from frame_id / frame_text, which stay.
EXCEL_HIDDEN = ["prompt_sent"]
EXCEL_COLUMNS = [c for c in RECORD_FIELDS if c not in EXCEL_HIDDEN]

# Back-compat alias: existing code that says config.COLUMNS means the record.
COLUMNS = RECORD_FIELDS
