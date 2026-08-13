"""
Streamlit dashboard for the cross-lingual ontology probe.

Runs one probe across all three languages and both models at once, so the
comparison the study is actually about — does the same question in Hindi get
the same ontological commitment as in English — is visible in one screen.

Every call goes through the same call_model() and the same rate limiter as the
batch runner, and lands in the same log.

    streamlit run dashboard.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

import config
import excel_logger
import nvidia_client as nv

st.set_page_config(
    page_title="Cross-lingual Ontology Probe",
    page_icon="◲",
    layout="wide",
)

LANG_LABEL = {"EN": "English", "HI": "हिन्दी  Hindi", "HG": "Hinglish"}
LANG_ORDER = ["EN", "HI", "HG"]

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1500px;}
      .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 55%, #3d7ea6 100%);
        color: #fff; padding: 1.6rem 1.9rem; border-radius: 14px; margin-bottom: 1.4rem;
      }
      .hero h1 {font-size: 1.75rem; margin: 0 0 .35rem 0; font-weight: 650; color:#fff;}
      .hero p  {margin: 0; opacity: .92; font-size: .95rem; line-height: 1.5;}
      .hero .rq {margin-top: .7rem; font-size: .87rem; opacity: .85; font-style: italic;}
      .pill {
        display:inline-block; padding:.16rem .6rem; border-radius:999px;
        font-size:.72rem; font-weight:600; letter-spacing:.02em; margin-right:.35rem;
      }
      .pill-ok   {background:#dcfce7; color:#15803d;}
      .pill-bad  {background:#fee2e2; color:#b91c1c;}
      .pill-warn {background:#fef3c7; color:#a16207;}
      .pill-lang {background:#e0e7ff; color:#3730a3;}
      .pill-arm  {background:#f3e8ff; color:#6b21a8;}
      .respbox {
        border:1px solid rgba(128,128,128,.22); border-radius:10px;
        padding:.9rem 1.05rem; background:rgba(250,250,252,.6);
        max-height:420px; overflow-y:auto; font-size:.9rem; line-height:1.55;
      }
      .meta {font-size:.75rem; opacity:.62; margin-top:.4rem;}
      .probe {
        font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
        background: rgba(128,128,128,.09); border-left: 3px solid #3d7ea6;
        padding: .55rem .8rem; border-radius: 6px; font-size: .87rem; margin-bottom:.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _bank():
    return nv.load_prompt_bank()


bank = _bank()
prompts = bank["prompts"]

# Group the bank by probe: one probe = one family + paraphrase, realised in
# three languages. That triple is the unit the dashboard runs.
PROBES: dict = {}
for p in prompts:
    PROBES.setdefault(f"{p['family']}{p['paraphrase']}", {})[p["language"]] = p


def probe_label(key: str) -> str:
    # No arm label here. The arm is recorded in the log for analysis, but it is
    # kept off the screen next to the prompt so that nobody reading a response
    # is primed by knowing which side of the pair they are looking at.
    trio = PROBES[key]
    en = trio.get("EN") or next(iter(trio.values()))
    return f"{key}  ·  {en['prompt_text'][:58]}"


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <h1>Cross-lingual Ontology Probe</h1>
      <p>Does a model treat <b>the family</b> as an entity in its own right, or as a way
      of speaking about individuals — and does that answer change when the identical
      question is asked in Hindi rather than English? Each probe is paired with a
      matched <b>company</b> control, so the question is not whether groups can be
      entities, but <i>which</i> groups are allowed to be.</p>
      <p class="rq">One call, one fresh session. No system prompt, no preamble.
      The prompt text is the entire user message.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Probe")

    mode = st.radio(
        "Prompt source",
        ["Probe bank", "Custom prompt"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Probe bank":
        probe_key = st.selectbox(
            "Probe", list(PROBES), format_func=probe_label, label_visibility="collapsed"
        )
        trio = PROBES[probe_key]
        languages = st.multiselect(
            "Languages",
            [l for l in LANG_ORDER if l in trio],
            default=[l for l in LANG_ORDER if l in trio],
            format_func=lambda l: LANG_LABEL[l],
            help="All three by default — the same question, three realisations. "
                 "This is the comparison the study is about.",
        )
        targets = [trio[l] for l in languages]
        custom_text = ""
    else:
        st.caption(
            "One box per language. Fill only the ones you want — an empty box "
            "is skipped. Fill all three with the same question to get the "
            "cross-lingual comparison on your own wording."
        )
        placeholders = {
            "EN": "Is a family a person?",
            "HI": "क्या परिवार एक व्यक्ति है?",
            "HG": "Kya parivaar ek vyakti hai?",
        }
        targets = []
        for lang in LANG_ORDER:
            text = st.text_area(
                LANG_LABEL[lang], height=90, key=f"custom_{lang}",
                placeholder=placeholders[lang],
            )
            if text.strip():
                targets.append({
                    "prompt_id": f"custom-{lang}",
                    "prompt_text": text.strip(),
                    "language": lang,
                    "family": "custom", "arm": "", "form": "", "paraphrase": "",
                })
        if targets:
            st.caption(f"{len(targets)} of 3 filled — "
                       f"{', '.join(t['language'] for t in targets)} will run.")
        custom_text = ""

    st.divider()
    st.markdown("### Models")
    model_names = st.multiselect(
        "Models", list(config.MODELS), default=list(config.MODELS),
        format_func=lambda m: f"{m}  ·  {config.MODELS[m]['rpm_limit']} rpm",
        label_visibility="collapsed",
        help="Called concurrently, so both models see the prompt in the same minute.",
    )

    st.divider()
    st.markdown("### Run")

    run_once = st.button("Run once", type="primary", width="stretch",
                         help="One call per language × model, at the provider default.")

    with st.expander("Temperature sweep"):
        temps = st.multiselect(
            "Temperatures", [0.0, 0.2, 0.5, 0.7, 1.0, 1.2, 1.4, 1.7, 2.0],
            default=config.SWEEP_TEMPERATURES,
        )
        sweep_reps = st.number_input("Replicates per temperature", 1, 8,
                                     config.SWEEP_REPLICATES)
        run_sweep = st.button("Run sweep", width="stretch")
        st.caption(f"{len(targets) * len(model_names) * len(temps) * int(sweep_reps)} "
                   f"calls. Logged as `condition=sweep` with the temperature on "
                   f"every row — kept apart from the protocol run.")

    with st.expander("Protocol run"):
        st.caption(
            f"{config.DEFAULT_REPLICATES} replicates at the provider default "
            f"+ {config.DETERMINISM_REPLICATES} at temperature 0 — the design "
            f"Appendix B calibrates n against. "
            f"{len(targets) * len(model_names) * (config.DEFAULT_REPLICATES + config.DETERMINISM_REPLICATES)} calls."
        )
        run_protocol = st.button("Run protocol", width="stretch")

    st.divider()
    st.caption(
        f"bank {bank['bank_version']} · {len(prompts)} prompts · "
        f"sha `{bank['sha256'][:10]}…` · max_tokens {config.MAX_TOKENS}"
    )
    if config.USE_BREVITY_FRAME:
        st.warning(
            f"Frame **`{config.BREVITY_FRAME_ID}`** active — a length instruction "
            f"is appended to every prompt in its own language. This is priming; "
            f"it is logged in `frame_id`/`frame_text` and its effect on the "
            f"assertion rate has to be reported. Off via `config.py`."
        )


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _one(target, model, condition, replicate_index, temp):
    rows = nv.call_model(
        model_name=model,
        prompt_text=target["prompt_text"],
        condition=condition,
        replicate_index=replicate_index,
        source="dashboard",
        prompt_meta=target,
        temperature_override=temp,
        bank_sha256=bank["sha256"],
        bank_version=bank["bank_version"],
        verbose=False,
    )
    excel_logger.append_rows(rows)
    return rows[-1]


def _fan_out(jobs, label):
    """Run every (target, model, ...) job concurrently. No Streamlit calls inside."""
    results = {}
    bar = st.progress(0.0, text=f"{label} · 0/{len(jobs)}")
    done = 0
    with ThreadPoolExecutor(max_workers=min(6, len(jobs))) as pool:
        futures = {pool.submit(_one, *j[1:]): j[0] for j in jobs}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            bar.progress(done / len(jobs), text=f"{label} · {done}/{len(jobs)}")
    bar.empty()
    return results


def _status_pill(row):
    if row["status"] != "success":
        return '<span class="pill pill-bad">failed</span>'
    if row["truncated"]:
        return '<span class="pill pill-warn">truncated</span>'
    return '<span class="pill pill-ok">success</span>'


def _render(row, target):
    st.markdown(
        f'{_status_pill(row)}'
        f'<span class="pill pill-lang">{LANG_LABEL.get(target["language"], "custom")}</span>',
        unsafe_allow_html=True,
    )
    if row["status"] == "success":
        st.markdown(f'<div class="respbox">{row["response_text"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="meta">{row["total_tokens"]} tok · '
            f'{row["latency_seconds"]}s · {row["response_chars"]} chars · '
            f'temp {row["temperature"]} · {row["model_id_returned"]}</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Verbatim (as logged)"):
            st.text(row["response_text"])
    else:
        st.error(row["response_text"][:400])


def _grid(results, targets, title):
    """One row per language, one column per model."""
    st.markdown(f"#### {title}")
    for target in targets:
        sent, fid, _ = nv.apply_frame(target["prompt_text"], target["language"])
        st.markdown(f'<div class="probe">{sent}</div>', unsafe_allow_html=True)
        cols = st.columns(len(model_names))
        for col, model in zip(cols, model_names):
            with col:
                st.markdown(f"**{model}**")
                key = (target["prompt_id"], model)
                if key in results:
                    _render(results[key], target)
        st.divider()


_ready = bool(targets) and bool(model_names)
if not _ready:
    st.info("Pick a probe and at least one model in the sidebar, then hit **Run once**.")

if run_once and _ready:
    # Replicate numbers are resolved here, in the main thread, before any
    # worker starts — two threads asking the log at once would both get 1.
    jobs = [
        ((t["prompt_id"], m), t, m, "default",
         excel_logger.next_replicate_index(t["prompt_id"], m, "default"), None)
        for t in targets for m in model_names
    ]
    res = _fan_out(jobs, "running")
    st.success(f"{len(res)} calls logged to `{config.EXCEL_LOG_PATH.name}`")
    _grid(res, targets, "Responses")

if run_sweep and _ready:
    if not temps:
        st.warning("Pick at least one temperature.")
    else:
        st.markdown("#### Temperature sweep")
        for temp in sorted(temps):
            jobs = []
            for t in targets:
                for m in model_names:
                    base = excel_logger.next_replicate_index(
                        t["prompt_id"], m, "sweep", temperature=temp
                    )
                    for r in range(int(sweep_reps)):
                        jobs.append(((t["prompt_id"], m), t, m, "sweep", base + r, temp))
            res = _fan_out(jobs, f"temperature {temp}")
            with st.expander(f"temperature = {temp}", expanded=(temp == sorted(temps)[0])):
                _grid(res, targets, f"T = {temp}")

if run_protocol and _ready:
    plan = nv.elicitation_plan()
    st.markdown("#### Protocol run")
    for slot in plan:
        jobs = [
            ((t["prompt_id"], m), t, m, slot["condition"],
             excel_logger.next_replicate_index(t["prompt_id"], m, slot["condition"]), None)
            for t in targets for m in model_names
        ]
        res = _fan_out(jobs, f"{slot['condition']} #{slot['replicate_index']}")
        with st.expander(f"{slot['condition']} #{slot['replicate_index']}",
                         expanded=(slot["condition"] == "determinism")):
            _grid(res, targets, f"{slot['condition']} #{slot['replicate_index']}")

# --------------------------------------------------------------------------
# Log
# --------------------------------------------------------------------------

st.divider()
s = excel_logger.summarise()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Calls logged", s["attempts"])
m2.metric("Success", s["success"])
m3.metric("Failed", s["failed"])
m4.metric("Truncated", s["truncated"])
m5.metric("Tokens", f"{s['total_tokens']:,}")

left, right = st.columns([4, 1])
left.markdown("#### Recent calls")
n_tail = right.number_input("rows", 5, 200, 15, step=5, label_visibility="collapsed")

tail = excel_logger.read_tail(int(n_tail))
if not tail:
    st.info("Nothing logged yet.")
else:
    cols = ["prompt_id", "language", "prompt_text", "prompt_sent",
            "model_name", "condition", "temperature", "replicate_index",
            "response_text", "status", "response_chars", "total_tokens",
            "timestamp"]
    df = pd.DataFrame(tail).reindex(columns=cols).astype(str)
    # Same order as the spreadsheet, and nothing is sliced. The full string is
    # in every cell — click one to expand it.
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            "prompt_text": st.column_config.TextColumn(
                "prompt_text", width="medium"),
            "prompt_sent": st.column_config.TextColumn(
                "prompt_sent (frame included)", width="medium"),
            "response_text": st.column_config.TextColumn(
                "response_text (click a cell to expand)", width="large"),
            "timestamp": st.column_config.TextColumn(width="small"),
        },
    )
    st.caption(
        "This table is a view of `run_log.jsonl`. The spreadsheet and the JSONL "
        "always hold the complete response — verify with the checker below."
    )

    with open(config.EXCEL_LOG_PATH, "rb") as fh:
        st.download_button("Download run_log.xlsx", fh, file_name="run_log.xlsx",
                           mime="application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet")
