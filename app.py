"""
app.py — Ampliflow.ai Memory Engine
Streamlit interface for the Sticky Memory Model and Strategy Evaluator
"""

import os
import streamlit as st
from memory_engine import AmpliflowMemoryEngine
from strategy_evaluator import (
    build_evaluator_graph,
    EvaluationState,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ampliflow.ai — Memory Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; color: #f0f0f0; }
    [data-testid="stSidebar"] { background-color: #141414; border-right: 1px solid #222; }

    .icp-badge {
        display: inline-block;
        background: #1a3a2a; border: 1px solid #2ecc71;
        color: #2ecc71; border-radius: 20px;
        padding: 4px 14px; font-size: 13px; font-weight: 600;
        margin-bottom: 12px;
    }
    .token-badge {
        display: inline-block;
        background: #1a2a3a; border: 1px solid #3498db;
        color: #3498db; border-radius: 20px;
        padding: 4px 14px; font-size: 13px; font-weight: 600;
        margin-left: 8px; margin-bottom: 12px;
    }
    .section-label {
        font-size: 11px; font-weight: 700;
        letter-spacing: 1.5px; color: #888;
        text-transform: uppercase; margin-bottom: 6px;
    }
    .chunk-source {
        font-size: 10px; font-weight: 700;
        letter-spacing: 1px; color: #2ecc71;
        text-transform: uppercase; margin: 10px 0 4px 0;
        border-top: 1px solid #222; padding-top: 8px;
    }
    .preview-box {
        background: #111; border: 1px solid #222;
        border-radius: 8px; padding: 14px;
        font-size: 13px; color: #ccc;
        line-height: 1.7; white-space: pre-wrap;
        max-height: 320px; overflow-y: auto;
    }
    .agent-card {
        background: #141414; border: 1px solid #2a2a2a;
        border-radius: 8px; padding: 12px 16px;
        margin-bottom: 10px;
    }
    .agent-label {
        font-size: 10px; font-weight: 700;
        letter-spacing: 1.5px; text-transform: uppercase;
        margin-bottom: 6px;
    }
    .agent-growth  { color: #2ecc71; }
    .agent-risk    { color: #e74c3c; }
    .agent-icp     { color: #3498db; }
    .agent-synth   { color: #f39c12; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cached engine loader (only runs once per session) ─────────────────────────
@st.cache_resource
def load_engine(user_id: str) -> AmpliflowMemoryEngine:
    engine = AmpliflowMemoryEngine(user_id=user_id)
    engine.load_session_state()
    engine.build_vectorstore()
    return engine


# ── Session state init ─────────────────────────────────────────────────────────
if "engine_ready" not in st.session_state:
    st.session_state.engine_ready = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "strategy_report" not in st.session_state:
    st.session_state.strategy_report = None
if "debate_log" not in st.session_state:
    st.session_state.debate_log = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Ampliflow.ai")
    st.markdown("**Memory Engine** — Internal Tool")
    st.divider()

    st.markdown("#### 👤 User Session")
    user_id = st.selectbox(
        "Select User",
        ["user_102"],
        help="In production this would be the logged-in user's ID"
    )

    if st.button("🔌 Load Memory", use_container_width=True, type="primary"):
        with st.spinner("Loading memory from disk..."):
            try:
                engine = load_engine(user_id)
                st.session_state.engine_ready = True
                st.success("Memory loaded")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.session_state.engine_ready:
        engine = load_engine(user_id)
        st.divider()
        st.markdown("#### 📊 Memory Status")

        state = engine.session_state
        last_active = state.get("last_active", "—")
        st.markdown("**Last Active:**")
        st.caption(last_active[:19].replace("T", " ") if last_active != "—" else "—")

        last_icp = state.get("last_icp_loaded", "—")
        st.markdown("**Last ICP:**")
        st.caption(
            last_icp.replace("icp_", "").replace(".md", "").replace("_", " ").title()
            if last_icp != "—" else "—"
        )

        last_query = state.get("last_query", "—")
        st.markdown("**Last Query:**")
        st.caption(last_query[:55] + "..." if len(last_query) > 55 else last_query)

        count = engine.vectorstore._collection.count()
        st.markdown("**Vector Store:**")
        st.caption(f"{count} chunks across 8 ICP profiles")

    st.divider()
    st.caption("Built for Banyan.Works / Ampliflow.ai")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧠 Memory Engine", "🎯 Strategy Evaluator"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — MEMORY ENGINE
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Content Generation Request")
    st.markdown(
        "Describe the content you need. The engine automatically identifies "
        "the most relevant customer profile and assembles the right context — "
        "ICP, brand voice and compliance rules — within your token budget."
    )

    if not st.session_state.engine_ready:
        st.info("👈 Load memory from the sidebar first.")
    else:
        query = st.text_area(
            "What content do you need to generate?",
            placeholder=(
                "e.g. Write a LinkedIn post announcing our new AI feature for B2B buyers\n"
                "e.g. Write a donor thank-you email for our charity's annual fundraising campaign\n"
                "e.g. Write a neighbourhood Facebook post for our local trades business"
            ),
            height=100,
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            run = st.button(
                "⚡ Assemble Context",
                type="primary",
                use_container_width=True,
                disabled=not query.strip()
            )

        if run and query.strip():
            with st.spinner("Retrieving ICP, assembling context..."):
                try:
                    engine = load_engine(user_id)
                    result = engine.assemble_context(query)
                    st.session_state.last_result = result
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.last_result = None

        if st.session_state.last_result:
            result = st.session_state.last_result
            budget = result["token_budget"]

            st.divider()

            # ── Badges ────────────────────────────────────────────────────────
            icp_display = (
                result["icp_source"]
                .replace("icp_", "")
                .replace(".md", "")
                .replace("_", " ")
                .title()
            )
            token_pct = int((budget["total"] / budget["limit"]) * 100)

            st.markdown(
                f'<span class="icp-badge">🎯 ICP: {icp_display}</span>'
                f'<span class="token-badge">'
                f'🪙 {budget["total"]} / {budget["limit"]} tokens ({token_pct}%)'
                f'</span>',
                unsafe_allow_html=True
            )

            # ── Token breakdown ───────────────────────────────────────────────
            with st.expander("Token breakdown by component", expanded=False):
                cols = st.columns(5)
                labels = ["System", "ICP", "Brand Voice", "Compliance", "Query"]
                keys   = ["system_prompt", "icp_context", "brand_voice", "compliance", "user_query"]
                for col, label, key in zip(cols, labels, keys):
                    col.metric(label, budget[key])

                st.progress(
                    budget["total"] / budget["limit"],
                    text=f"{budget['remaining']} tokens remaining for generation"
                )

            # ── Three column context display ───────────────────────────────────
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    '<div class="section-label">ICP Profile Retrieved</div>',
                    unsafe_allow_html=True
                )
                # Split chunks by double newline and label each with its source
                chunks = result["icp_context"].split("\n\n")
                icp_html = ""
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        icp_html += (
                            f'<div class="chunk-source">Chunk {i+1} — '
                            f'{result["icp_source"].replace("icp_","").replace(".md","").replace("_"," ").title()}'
                            f'</div>{chunk.strip()}\n\n'
                        )
                st.markdown(
                    f'<div class="preview-box">{icp_html}</div>',
                    unsafe_allow_html=True
                )

            with c2:
                st.markdown(
                    '<div class="section-label">Brand Voice</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="preview-box">{result["brand_voice"][:800]}'
                    f'{"..." if len(result["brand_voice"]) > 800 else ""}</div>',
                    unsafe_allow_html=True
                )

            with c3:
                st.markdown(
                    '<div class="section-label">Compliance Rules Loaded</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="preview-box">{result["compliance"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── System prompt ─────────────────────────────────────────────────
            st.divider()
            st.markdown(
                '<div class="section-label">Assembled System Prompt</div>',
                unsafe_allow_html=True
            )
            st.code(result["system_prompt"], language=None)

            st.success(
                f"✅ Context ready. {budget['remaining']} tokens remaining "
                f"for content generation output."
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — STRATEGY EVALUATOR
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Business Strategy Evaluator")
    st.markdown(
        "Submit a business strategy in plain English. Three AI personas debate it: "
        "a **Growth Champion** finds the upside, a **Risk Challenger** finds the failure points, "
        "and an **ICP Simulator** stress-tests it as the target buyer. "
        "A synthesizer then produces a structured report with SWOT, risk index, and go/no-go."
    )

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        st.warning(
            "⚠️ GROQ_API_KEY not set. "
            "In the terminal run: `set GROQ_API_KEY=your_key_here` then restart Streamlit."
        )

    strategy_input = st.text_area(
        "Describe your business strategy",
        placeholder=(
            "e.g. We are launching a B2B SaaS platform that helps content marketing "
            "teams generate targeted content using AI. Target customers are marketing "
            "managers at SMEs with 10-200 employees. Pricing is £99/month per seat. "
            "We have a working MVP, 3 beta customers, and 6 months of runway."
        ),
        height=160,
    )

    rounds = st.slider(
        "Debate rounds before synthesis",
        min_value=1,
        max_value=3,
        value=1,
        help="1 round is fastest for demos (~10s). 2 rounds gives a richer report."
    )

    run_strategy = st.button(
        "🎯 Run Evaluation",
        type="primary",
        disabled=not strategy_input.strip() or not groq_key
    )

    if run_strategy and strategy_input.strip():
        # Clear previous results
        st.session_state.strategy_report = None
        st.session_state.debate_log = []

        agent_colors = {
            "GROWTH CHAMPION": "agent-growth",
            "RISK CHALLENGER":  "agent-risk",
            "ICP SIMULATOR":    "agent-icp",
            "SYNTHESIZER":      "agent-synth",
        }

        debate_placeholder = st.empty()
        status_placeholder = st.empty()

        total_steps = rounds * 3 + 1
        progress = st.progress(0, text="Initialising debate...")

        try:
            graph = build_evaluator_graph()
            initial_state: EvaluationState = {
                "raw_strategy_input": strategy_input,
                "current_round": 1,
                "max_rounds": rounds,
                "debate_transcript": [],
                "risk_matrix": {},
                "final_report": "",
            }

            step = 0
            final_report = ""

            # Stream graph execution node by node
            for event in graph.stream(initial_state):
                for node_name, node_output in event.items():

                    step += 1
                    pct = min(int((step / total_steps) * 100), 95)
                    progress.progress(pct, text=f"Running: {node_name.replace('_', ' ').title()}...")

                    # Extract what this agent just said
                    new_msgs = node_output.get("debate_transcript", [])
                    for msg in new_msgs:
                        content = msg.content if hasattr(msg, "content") else str(msg)

                        # Parse agent label
                        agent_label = "AGENT"
                        agent_class = ""
                        for label, css_class in agent_colors.items():
                            if content.startswith(f"[{label}]"):
                                agent_label = label
                                agent_class = css_class
                                content = content[len(f"[{label}]"):].strip()
                                break

                        if agent_label != "SYNTHESIZER":
                            st.session_state.debate_log.append({
                                "label": agent_label,
                                "class": agent_class,
                                "content": content,
                            })

                    # Capture final report
                    if node_output.get("final_report"):
                        final_report = node_output["final_report"]

                    # Re-render debate log live
                    with debate_placeholder.container():
                        if st.session_state.debate_log:
                            st.markdown("#### 🗣️ Live Debate")
                            for entry in st.session_state.debate_log:
                                st.markdown(
                                    f'<div class="agent-card">'
                                    f'<div class="agent-label {entry["class"]}">'
                                    f'{entry["label"]}</div>'
                                    f'{entry["content"]}'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

            progress.progress(100, text="Complete ✓")
            st.session_state.strategy_report = final_report

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            progress.empty()

    # ── Display report ─────────────────────────────────────────────────────────
    if st.session_state.strategy_report:
        st.divider()

        # Show debate log collapsed if we have it
        if st.session_state.debate_log:
            with st.expander("📜 Full Debate Transcript", expanded=False):
                for entry in st.session_state.debate_log:
                    st.markdown(
                        f'<div class="agent-card">'
                        f'<div class="agent-label {entry["class"]}">{entry["label"]}</div>'
                        f'{entry["content"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        st.markdown("### 📋 Strategy Evaluation Report")
        st.markdown(st.session_state.strategy_report)

        st.divider()
        st.download_button(
            label="⬇️ Download Report as Markdown",
            data=st.session_state.strategy_report,
            file_name="ampliflow_strategy_report.md",
            mime="text/markdown",
        )