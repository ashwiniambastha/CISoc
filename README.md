# Cross-lingual LLM ontology probe — runner

Kinship bank (Appendix C v0.9): 36 prompts × 2 models × 9 elicitations = **648 elicitations**.

## Setup

```bash
pip install -r requirements.txt
```

`.env` already holds `NVIDIA_API_KEY`. Nothing else is read from the environment.

## Order of operations

```bash
python nvidia_client.py                       # 1. smoke test: one call per model
python excel_logger.py                        # 2. writes one test row, prints it back
python batch_runner.py --dry-run              # 3. see the plan, no calls
python batch_runner.py --prompts 2 --slots 1-2   # 4. tiny pilot, inspect by hand
python batch_runner.py --slots 1-4            # 5. replicates 1-4 today
python batch_runner.py --slots 5-9            #    the rest on another day
streamlit run dashboard.py                    # 6. manual one-offs
```

Delete the `source="self_test"` row from `logs/run_log.jsonl` and `run_log.xlsx`
before analysis, or filter on `source` — it is there to prove the logger works.

## Files

| File | Role |
|---|---|
| `prompt_bank.json` | Authoritative bank. NFC, hashed SHA-256 before the first call. Frozen. |
| `config.py` | Models, RPM caps, max_tokens, retry policy, log schema. One place to record. |
| `nvidia_client.py` | `RateLimiter`, `call_model()`. **Both** runner and dashboard go through it. |
| `excel_logger.py` | Append to JSONL (authoritative) then Excel. Resume index, summaries. |
| `batch_runner.py` | Full matrix, randomised order, resume, version-pin halt. |
| `dashboard.py` | Streamlit: pick model/temp/prompt, run once or all 9, live log tail. |
| `logs/run_log.xlsx` | Analysis sheet, one row per **attempt**. |
| `logs/run_log.jsonl` | Append-only mirror. Full untruncated text lives here. |
| `logs/run_manifest.json` | Written at the start of every batch run. |

## Decisions baked in

- **`max_tokens = 2048`**, identical across both models, recorded on every row.
  `finish_reason == "length"` sets `truncated = TRUE`; a truncated response is
  not a complete answer.
- **Retry once** on transport failure, then log `failed` and move on. Both
  attempts get their own row — the retry never overwrites the failure it
  followed.
- **Two rate-limit layers**: per-model rolling 60s window (40 rpm nemotron,
  30 rpm gpt_oss) *and* a flat 15s pause every 20 calls.
- **`default` condition sends no temperature and no top_p at all** — the
  provider default applies rather than a hand-set value that happens to match.
  Read the provider's defaults on the run date and paste them into
  `run_manifest.json` → `provider_defaults_on_run_date`.
- **Order** is re-randomised per replicate from `ORDER_SEED_BASE`; no two
  consecutive prompts share a family or language where avoidable. The seed is
  on every row.
- **Resume** is automatic: a `(prompt_id, model, condition, replicate)` with a
  successful `source=batch` row is skipped. `--no-resume` overrides.
- **Version pinning**: if `model_id_returned` ever differs from what was sent,
  the run halts (C.7.6) rather than pooling across the boundary.

## Before the full run

1. Confirm the real RPM numbers on your build.nvidia.com dashboard and update
   `config.MODELS[...]["rpm_limit"]`. 40/30 are typical, not guaranteed.
2. Run the smoke test and check `model_id_returned` matches `model_id` exactly
   for both models — that string is what pins the version.
3. Check a Hindi and a Hinglish response render correctly in Excel (UTF-8
   round-trip) before committing 648 calls.
4. Fill in `provider_defaults_on_run_date` in the manifest.
