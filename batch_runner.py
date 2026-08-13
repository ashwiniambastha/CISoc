"""
Unattended runner for the full matrix.

    36 prompts x 2 models x 9 elicitations = 648 elicitations

Ordering follows Appendix C.7.2: the prompt order is re-randomised for every
replicate, the seed is recorded on every row, and no two consecutive prompts
share a family, arm or language where that can be avoided.

Usage
-----
    python batch_runner.py --dry-run                  # plan only, no API calls
    python batch_runner.py --prompts 2 --slots 2      # small pilot, inspect by hand
    python batch_runner.py --slots 1-4                # replicates 1-4 today
    python batch_runner.py --slots 5-9                # the rest tomorrow
    python batch_runner.py                            # everything not yet done
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import config
import excel_logger
import nvidia_client as nv


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------


def spaced_order(prompts: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """Shuffle, then greedily avoid consecutive same family / arm / language."""
    pool = list(prompts)
    random.Random(seed).shuffle(pool)

    ordered: List[Dict[str, Any]] = []
    prev: Dict[str, Any] = {}
    while pool:
        pick = None
        for strictness in ("family+lang", "family", "any"):
            for cand in pool:
                if strictness == "family+lang":
                    ok = (
                        cand.get("family") != prev.get("family")
                        and cand.get("language") != prev.get("language")
                    )
                elif strictness == "family":
                    ok = cand.get("family") != prev.get("family")
                else:
                    ok = True
                if ok:
                    pick = cand
                    break
            if pick is not None:
                break
        pool.remove(pick)
        ordered.append(pick)
        prev = pick
    return ordered


def parse_slots(spec: str) -> List[Dict[str, Any]]:
    """Turn '1-4' / '3' / 'all' into a list of {condition, replicate_index}."""
    plan = nv.elicitation_plan()  # 8 default + 1 determinism, in order
    if spec in ("", "all", None):
        return plan
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-", 1))
    else:
        lo = hi = int(spec)
    return plan[lo - 1 : hi]


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def write_manifest(bank: Dict[str, Any], args: argparse.Namespace, slots) -> None:
    import openai

    manifest = {
        "run_started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bank_version": bank["bank_version"],
        "bank_sha256": bank["sha256"],
        "n_prompts": len(bank["prompts"]),
        "provider": "NVIDIA build.nvidia.com",
        "base_url": config.BASE_URL,
        "models": config.MODELS,
        "max_tokens": config.MAX_TOKENS,
        "default_condition": "temperature/top_p unset - provider default applies; "
                             "read the defaults off the provider docs on the run date "
                             "and record them here by hand",
        "provider_defaults_on_run_date": "<fill in from provider docs>",
        "response_length_frame": {
            "in_use": config.USE_BREVITY_FRAME,
            "frame_id": config.BREVITY_FRAME_ID if config.USE_BREVITY_FRAME else "none",
            "placement": config.BREVITY_FRAME_PLACEMENT,
            "text_by_language": config.BREVITY_FRAMES if config.USE_BREVITY_FRAME else {},
            "status": "This is priming, added deliberately (Appendix B.3). Its effect "
                      "on the assertion rate must be reported. Do not pool rows "
                      "carrying different frame_id values.",
        },
        "determinism_condition": {
            "temperature": config.DETERMINISM_TEMPERATURE,
            "top_p": config.DETERMINISM_TOP_P,
            "seed": config.DETERMINISM_SEED,
        },
        "retry_policy": f"retry once on transport failure "
                        f"(MAX_ATTEMPTS={config.MAX_ATTEMPTS}), every attempt logged",
        "rate_limiting": {
            "per_model_rpm": {k: v["rpm_limit"] for k, v in config.MODELS.items()},
            "calls_before_pause": config.CALLS_BEFORE_PAUSE,
            "pause_seconds": config.PAUSE_SECONDS,
        },
        "order_seed_base": config.ORDER_SEED_BASE,
        "slots_this_run": slots,
        "system_prompt": None,
        "tools_retrieval_browsing": "disabled (not sent)",
        "openai_client_version": getattr(openai, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "cli_args": vars(args),
    }
    config.MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"manifest -> {config.MANIFEST_PATH}")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-lingual ontology probe batch runner")
    ap.add_argument("--slots", default="all",
                    help="which of the 9 elicitations to run, e.g. '1-4', '9', 'all'")
    ap.add_argument("--prompts", type=int, default=0,
                    help="cap the number of prompts (0 = whole bank); for pilots")
    ap.add_argument("--models", default="",
                    help="comma-separated subset, e.g. 'nemotron'")
    ap.add_argument("--languages", default="",
                    help="comma-separated subset, e.g. 'EN,HI'")
    ap.add_argument("--families", default="",
                    help="comma-separated subset, e.g. 'K-K1,K-K2'")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without calling the API")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-elicit even where a successful row already exists")
    args = ap.parse_args()

    bank = nv.load_prompt_bank()
    prompts = bank["prompts"]

    if args.languages:
        wanted = {x.strip().upper() for x in args.languages.split(",")}
        prompts = [p for p in prompts if p["language"] in wanted]
    if args.families:
        wanted = {x.strip() for x in args.families.split(",")}
        prompts = [p for p in prompts if p["family"] in wanted]
    if args.prompts:
        prompts = prompts[: args.prompts]

    models = [m.strip() for m in args.models.split(",") if m.strip()] or list(config.MODELS)
    for m in models:
        if m not in config.MODELS:
            sys.exit(f"Unknown model {m!r}. Choose from {list(config.MODELS)}")

    slots = parse_slots(args.slots)
    done = set() if args.no_resume else excel_logger.completed_keys()

    # Build the work list.
    work: List[Dict[str, Any]] = []
    for slot in slots:
        seed = config.ORDER_SEED_BASE + slot["replicate_index"] * 1000 + (
            0 if slot["condition"] == "default" else 7
        )
        for prompt in spaced_order(prompts, seed):
            for model_name in models:
                key = (prompt["prompt_id"], model_name, slot["condition"],
                       slot["replicate_index"])
                if key in done:
                    continue
                work.append({"prompt": prompt, "model": model_name,
                             "slot": slot, "seed": seed})

    total_planned = len(prompts) * len(models) * len(slots)
    print(f"bank {bank['bank_version']}  sha256={bank['sha256'][:16]}...")
    print(f"{len(prompts)} prompts x {len(models)} models x {len(slots)} elicitations "
          f"= {total_planned} calls planned")
    print(f"{total_planned - len(work)} already logged, {len(work)} to run")
    if not work:
        print("nothing to do.")
        return

    slowest_rpm = min(config.MODELS[m]["rpm_limit"] for m in models)
    est_min = len(work) / max(slowest_rpm, 1) + (
        len(work) // config.CALLS_BEFORE_PAUSE * config.PAUSE_SECONDS / 60
    )
    print(f"rough floor on wall-clock: ~{est_min:.0f} min (rate limits + pauses)\n")

    if args.dry_run:
        for i, item in enumerate(work[:20], 1):
            s = item["slot"]
            print(f"  {i:>3}. {item['prompt']['prompt_id']:<10} {item['model']:<9} "
                  f"{s['condition']}#{s['replicate_index']}")
        if len(work) > 20:
            print(f"  ... and {len(work) - 20} more")
        print("\n(dry run: no calls made)")
        return

    write_manifest(bank, args, [f"{s['condition']}#{s['replicate_index']}" for s in slots])
    excel_logger.ensure_workbook()

    started = time.time()
    n_ok = n_fail = 0
    try:
        for i, item in enumerate(work, 1):
            prompt, model_name, slot = item["prompt"], item["model"], item["slot"]
            rows = nv.call_model(
                model_name=model_name,
                prompt_text=prompt["prompt_text"],
                condition=slot["condition"],
                replicate_index=slot["replicate_index"],
                source="batch",
                prompt_meta=prompt,
                bank_sha256=bank["sha256"],
                bank_version=bank["bank_version"],
                order_seed=item["seed"],
            )
            excel_logger.append_rows(rows)  # every attempt, success or failure

            final = rows[-1]
            if final["status"] == "success":
                n_ok += 1
            else:
                n_fail += 1

            elapsed = time.time() - started
            eta = elapsed / i * (len(work) - i) / 60
            print(f"[{i}/{len(work)}] {prompt['prompt_id']:<10} {model_name:<9} "
                  f"{slot['condition']}#{slot['replicate_index']:<2} "
                  f"{final['status']:<7} {str(final['total_tokens']):>5} tok  "
                  f"eta {eta:.0f}m")

            # Version pinning (Appendix C.7.6): halt if the model string moves.
            returned = final.get("model_id_returned") or ""
            expected = config.MODELS[model_name]["model_id"]
            if final["status"] == "success" and returned and returned != expected:
                print(f"\n!! model_id changed for {model_name}: expected {expected}, "
                      f"got {returned}. Halting per C.7.6 -- do not pool across "
                      f"the boundary.")
                break

    except KeyboardInterrupt:
        print("\ninterrupted -- everything up to this point is logged; "
              "re-run to resume.")

    print(f"\ndone: {n_ok} success, {n_fail} failed, "
          f"{(time.time() - started) / 60:.1f} min")
    print(json.dumps(excel_logger.summarise(), indent=2))
    print(f"log -> {config.EXCEL_LOG_PATH}")


if __name__ == "__main__":
    main()
