"""
Streamlit dashboard for manual, one-off elicitations.

Runs through the same call_model() and the same rate limiter as the batch
runner, and appends to the same log, distinguished only by source='dashboard'.

    streamlit run dashboard.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

import config
import excel_logger
import nvidia_client as nv

st.set_page_config(page_title="Ontology Probe Runner", layout="wide")


@st.cache_data(show_spinner=False)
def _bank():
    return nv.load_prompt_bank()


bank = _bank()
prompts = bank["prompts"]

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Run a probe")

    model_names = st.multiselect(
        "Models",
        list(config.MODELS),
        default=list(config.MODELS),
        format_func=lambda m: f"{m}  ({config.MODELS[m]['rpm_limit']} rpm)",
        help="Both by default. They are called concurrently, so the two models "
             "see the same prompt in the same minute rather than minutes apart.",
    )
    for m in model_names:
        st.caption(config.MODELS[m]["model_id"])
    if not model_names:
        st.warning("Pick at least one model.")

    st.divider()

    unset_temp = st.checkbox(
        "Leave temperature unset (provider default)",
        value=True,
        help="This is the protocol's `default` condition. Uncheck to hand-set a "
             "value; the row is then logged as condition='manual'.",
    )
    temperature = st.slider(
        "Temperature", 0.0, 2.0, 0.2, 0.05, disabled=unset_temp
    )

    st.divider()

    lang = st.radio("Language", ["All", "EN", "HI", "HG"], horizontal=True)
    pool = prompts if lang == "All" else [p for p in prompts if p["language"] == lang]

    source_choice = st.radio("Prompt source", ["Pick from prompt bank", "Type custom prompt"])

    if source_choice == "Pick from prompt bank":
        chosen = st.selectbox(
            "Prompt",
            pool,
            format_func=lambda p: f"{p['prompt_id']} — {p['prompt_text'][:60]}",
        )
        prompt_text = chosen["prompt_text"]
        prompt_meta = chosen
    else:
        prompt_text = st.text_area("Custom prompt", height=120,
                                   placeholder="Sent verbatim as the entire user message")
        prompt_meta = {"prompt_id": "custom", "language": lang if lang != "All" else "",
                       "family": "", "arm": "", "form": "", "paraphrase": ""}

    st.divider()
    run_one = st.button("Run once", type="primary", width="stretch")
    run_nine = st.button("Run all 9 elicitations", width="stretch",
                         help="8 at provider default + 1 at temperature 0, "
                              "through the same rate limiter as the batch runner")
    

    if config.USE_BREVITY_FRAME:
        st.warning(
           ""
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _run(model: str, condition: str, replicate_index: int, temp_override):
    """One elicitation on one model. Safe to call from a worker thread:
    it touches no Streamlit APIs, only the client and the logger."""
    rows = nv.call_model(
        model_name=model,
        prompt_text=prompt_text,
        condition=condition,
        replicate_index=replicate_index,
        source="dashboard",
        prompt_meta=prompt_meta,
        temperature_override=temp_override,
        bank_sha256=bank["sha256"],
        bank_version=bank["bank_version"],
        verbose=False,
    )
    excel_logger.append_rows(rows)
    return rows


def _run_across_models(condition: str, replicate_index: int, temp_override):
    """Fire every selected model at once. Returns {model: rows}."""
    if len(model_names) == 1:
        m = model_names[0]
        return {m: _run(m, condition, replicate_index, temp_override)}
    out = {}
    with ThreadPoolExecutor(max_workers=len(model_names)) as pool:
        futures = {
            pool.submit(_run, m, condition, replicate_index, temp_override): m
            for m in model_names
        }
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return {m: out[m] for m in model_names if m in out}  # stable order


def _show(row):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", row["status"])
    c2.metric("Prompt tokens", row["prompt_tokens"] or "—")
    c3.metric("Completion", row["completion_tokens"] or "—")
    c4.metric("Total", row["total_tokens"] or "—")
    c5.metric("Latency", f"{row['latency_seconds']}s" if row["latency_seconds"] != "" else "—")

    if row["status"] == "success":
        if row["truncated"]:
            st.warning("finish_reason = length: the response hit max_tokens and is "
                       "truncated. Not a complete answer.")
        st.markdown("**Response**")
        with st.container(border=True):
            # Render as markdown, which is what the models emit. Do NOT wrap
            # this in a pre-wrap div: that preserves the literal newlines on
            # top of the rendered structure and double-spaces everything.
            st.markdown(row["response_text"])
        with st.expander("Verbatim text (what is in the log)"):
            st.text(row["response_text"])
        st.caption(f"model_id_returned: {row['model_id_returned']} · "
                   f"finish_reason: {row['finish_reason']} · attempt {row['attempt']} · "
                   f"{row['response_chars']} chars")
    else:
        st.error(row["response_text"])


# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------

st.title("Cross-lingual ontology probe")
st.caption("One call, one fresh session. No system prompt, no preamble — "
           "the prompt text is the entire user message.")

if prompt_text:
    _sent, _fid, _ftext = nv.apply_frame(prompt_text, prompt_meta.get("language", ""))
    st.code(_sent, language=None)
    if _fid != "none":
        st.caption(f"frame_id `{_fid}` — the last line is appended, not part of the bank.")
    elif config.USE_BREVITY_FRAME:
        st.caption("No frame applied: this prompt has no language tag.")

_ready = bool(prompt_text.strip()) and bool(model_names)

if run_one:
    if not _ready:
        st.warning("Need a prompt and at least one model.")
    else:
        with st.spinner(f"calling {len(model_names)} model(s) in parallel…"):
            results = _run_across_models(
                condition="default" if unset_temp else "manual",
                replicate_index=1,
                temp_override=None if unset_temp else temperature,
            )
        n_rows = sum(len(r) for r in results.values())
        st.success(f"Saved to {config.EXCEL_LOG_PATH.name} ✅  "
                   f"({n_rows} row(s) written, source=dashboard)")

        # Side by side, so the two answers to the identical prompt can be read
        # against each other rather than one after the other.
        for col, model in zip(st.columns(len(results)), results):
            with col:
                st.subheader(model)
                _show(results[model][-1])

if run_nine:
    if not _ready:
        st.warning("Need a prompt and at least one model.")
    else:
        plan = nv.elicitation_plan()
        total = len(plan) * len(model_names)
        bar = st.progress(0.0, text=f"0/{total}")
        per_model = {m: [] for m in model_names}
        for i, slot in enumerate(plan, 1):
            batch = _run_across_models(
                condition=slot["condition"],
                replicate_index=slot["replicate_index"],
                temp_override=None,
            )
            for m, rows in batch.items():
                per_model[m].append(rows[-1])
            bar.progress(i / len(plan),
                         text=f"{i * len(model_names)}/{total}  "
                              f"{slot['condition']}#{slot['replicate_index']}")

        ok = sum(1 for rs in per_model.values() for r in rs if r["status"] == "success")
        st.success(f"Saved to {config.EXCEL_LOG_PATH.name} ✅  {ok}/{total} succeeded")

        for slot_i, slot in enumerate(plan):
            label = f"{slot['condition']} #{slot['replicate_index']}"
            states = " · ".join(
                f"{m}: {per_model[m][slot_i]['status']}" for m in model_names
            )
            with st.expander(f"{label} — {states}",
                             expanded=(slot["condition"] == "determinism")):
                for col, m in zip(st.columns(len(model_names)), model_names):
                    with col:
                        st.subheader(m)
                        _show(per_model[m][slot_i])

# --- log tail -------------------------------------------------------------

st.divider()
left, right = st.columns([3, 1])
left.subheader("Recent log rows")
n_tail = right.number_input("rows", 5, 200, 15, step=5, label_visibility="collapsed")

tail = excel_logger.read_tail(int(n_tail))
if not tail:
    st.info("No calls logged yet.")
else:
    cols = ["timestamp", "source", "prompt_id", "language", "model_name", "condition",
            "replicate_index", "temperature", "status", "total_tokens", "response_text"]
    df = pd.DataFrame(tail).reindex(columns=cols)
    # `temperature` is deliberately mixed: floats for set values, the string
    # "unset" for the provider-default condition. Arrow cannot type that column,
    # so everything is rendered as text.
    df = df.astype(str)
    df["response_text"] = df["response_text"].str.slice(0, 160)
    st.dataframe(df, width="stretch", hide_index=True)

    s = excel_logger.summarise()
    st.caption(
        f"log total: {s['attempts']} attempts · {s['success']} success · "
        f"{s['failed']} failed · {s['truncated']} truncated · "
        f"{s['total_tokens']:,} tokens"
    )
    with open(config.EXCEL_LOG_PATH, "rb") as fh:
        st.download_button("Download run_log.xlsx", fh, file_name="run_log.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
