"""
strategy_evaluator.py — Multi-Agent Business Strategy Evaluator
Uses LangGraph to run a structured debate between competing AI personas.
Outputs a SWOT report with risk index and go/no-go recommendation.
"""

import os
import operator
import logging
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── LLM setup ─────────────────────────────────────────────────────────────────
# Uses Groq API — fast, free tier available, no local GPU needed
# Set your key: set GROQ_API_KEY=your_key_here  (Windows CMD)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not set. Run: set GROQ_API_KEY=your_key_here"
    )

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.7,
)


# ── State definition ───────────────────────────────────────────────────────────
class EvaluationState(TypedDict):
    raw_strategy_input: str
    current_round: int
    max_rounds: int
    debate_transcript: Annotated[Sequence[BaseMessage], operator.add]
    risk_matrix: dict
    final_report: str


# ── Agent node helpers ─────────────────────────────────────────────────────────

def _get_transcript_text(state: EvaluationState) -> str:
    """Flatten debate transcript to plain text for context."""
    lines = []
    for msg in state["debate_transcript"]:
        role = "USER" if isinstance(msg, HumanMessage) else "AGENT"
        lines.append(f"[{role}] {msg.content}")
    return "\n".join(lines) if lines else "No prior debate yet."


def _call_agent(system_prompt: str, user_message: str) -> str:
    """Call the LLM with a system + user message pair."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"input": user_message})
    return response.content


# ── Agent nodes ────────────────────────────────────────────────────────────────

def growth_champion(state: EvaluationState) -> dict:
    """
    Optimistic growth analyst.
    Finds TAM, viral loops, network effects, scaling vectors.
    """
    logger.info(f"[Round {state['current_round']}] Growth Champion speaking...")

    system = """You are a growth-obsessed venture analyst. Your job is to find 
the most compelling case FOR this business strategy. Focus on:
- Total Addressable Market size and capture potential
- Viral loops, network effects, or compounding growth mechanisms  
- Timing advantages and market tailwinds
- Scalability vectors and unit economics potential
Be specific, be bullish, cite realistic analogies. Keep it to 150 words."""

    transcript = _get_transcript_text(state)
    user_msg = f"""Strategy to evaluate:
{state['raw_strategy_input']}

Prior debate:
{transcript}

Make the strongest growth case."""

    response = _call_agent(system, user_msg)
    logger.info("Growth Champion done.")

    return {
        "debate_transcript": [AIMessage(content=f"[GROWTH CHAMPION] {response}")],
        "current_round": state["current_round"],
    }


def risk_challenger(state: EvaluationState) -> dict:
    """
    Cynical CFO persona.
    Finds burn rate issues, churn risk, weak defensibility.
    """
    logger.info(f"[Round {state['current_round']}] Risk Challenger speaking...")

    system = """You are a battle-hardened CFO and risk analyst. Your job is to 
stress-test this strategy and find every way it could fail. Focus on:
- Burn rate and runway assumptions
- Churn risk and customer retention weaknesses
- Competitive moat or lack thereof
- Execution risk and team/resource gaps
- Regulatory or market timing risks
Be specific and ruthless. No empty encouragement. Keep it to 150 words."""

    transcript = _get_transcript_text(state)
    user_msg = f"""Strategy to evaluate:
{state['raw_strategy_input']}

Prior debate (including growth champion's case):
{transcript}

Identify the critical failure points."""

    response = _call_agent(system, user_msg)
    logger.info("Risk Challenger done.")

    return {
        "debate_transcript": [AIMessage(content=f"[RISK CHALLENGER] {response}")],
        "current_round": state["current_round"],
    }


def icp_simulator(state: EvaluationState) -> dict:
    """
    Simulates the target B2B buyer.
    Questions onboarding friction, pricing, and switching cost.
    """
    logger.info(f"[Round {state['current_round']}] ICP Simulator speaking...")

    system = """You are a sceptical B2B procurement manager evaluating whether 
to buy this product or service. You have budget authority and have been burned 
before. Ask the hard questions a real buyer would ask:
- Is the onboarding too complex for my team?
- Is the pricing model aligned with how we actually buy?
- What is my switching cost if this fails?
- Why would I trust this over an established alternative?
Speak in first person as the buyer. Keep it to 150 words."""

    transcript = _get_transcript_text(state)
    user_msg = f"""Strategy to evaluate:
{state['raw_strategy_input']}

Debate so far:
{transcript}

As the target buyer, raise your real objections."""

    response = _call_agent(system, user_msg)
    logger.info("ICP Simulator done.")

    return {
        "debate_transcript": [AIMessage(content=f"[ICP SIMULATOR] {response}")],
        "current_round": state["current_round"] + 1,
    }


def executive_synthesizer(state: EvaluationState) -> dict:
    """
    Reads the full debate transcript and outputs a structured strategy report.
    """
    logger.info("Executive Synthesizer producing final report...")

    system = """You are a senior strategy consultant producing a final evaluation 
report. Read the full debate transcript and synthesise it into a structured 
Markdown report with exactly these sections:

## Executive Summary
One paragraph overview of the strategy and evaluation outcome.

## SWOT Analysis
**Strengths:** (3 bullet points)
**Weaknesses:** (3 bullet points)  
**Opportunities:** (3 bullet points)
**Threats:** (3 bullet points)

## Risk Index
Score from 0 (no risk) to 10 (extreme risk) with one sentence justification.
Format: **Risk Index: X/10** — [justification]

## Recommendation
**GO** or **NO-GO** — followed by 2-3 sentences of reasoning.

## Top 3 Action Points
Numbered list of the three most important next steps if proceeding.

Be decisive. No hedging. This is a real recommendation."""

    transcript = _get_transcript_text(state)
    user_msg = f"""Original strategy:
{state['raw_strategy_input']}

Full debate transcript:
{transcript}

Produce the final structured report."""

    response = _call_agent(system, user_msg)
    logger.info("Final report generated.")

    return {
        "final_report": response,
        "debate_transcript": [AIMessage(content=f"[SYNTHESIZER] {response}")],
    }


# ── Routing logic ──────────────────────────────────────────────────────────────

def should_continue(state: EvaluationState) -> str:
    """Route back for another debate round or proceed to synthesis."""
    if state["current_round"] < state["max_rounds"]:
        return "continue"
    return "synthesize"


# ── Graph construction ─────────────────────────────────────────────────────────

def build_evaluator_graph() -> StateGraph:
    graph = StateGraph(EvaluationState)

    # Add nodes
    graph.add_node("growth_champion", growth_champion)
    graph.add_node("risk_challenger", risk_challenger)
    graph.add_node("icp_simulator", icp_simulator)
    graph.add_node("executive_synthesizer", executive_synthesizer)

    # Entry point
    graph.set_entry_point("growth_champion")

    # Linear debate flow within each round
    graph.add_edge("growth_champion", "risk_challenger")
    graph.add_edge("risk_challenger", "icp_simulator")

    # After ICP simulator: loop or synthesize
    graph.add_conditional_edges(
        "icp_simulator",
        should_continue,
        {
            "continue": "growth_champion",
            "synthesize": "executive_synthesizer",
        }
    )

    graph.add_edge("executive_synthesizer", END)

    return graph.compile()


# ── Public interface ───────────────────────────────────────────────────────────

def evaluate_strategy(strategy_text: str, rounds: int = 2) -> str:
    """
    Run the multi-agent strategy evaluation.
    
    Args:
        strategy_text: Plain English description of the business strategy.
        rounds: Number of debate rounds before synthesis (default 2).
    
    Returns:
        Final structured Markdown report as a string.
    """
    logger.info(f"Starting strategy evaluation ({rounds} rounds)...")

    initial_state: EvaluationState = {
        "raw_strategy_input": strategy_text,
        "current_round": 1,
        "max_rounds": rounds,
        "debate_transcript": [],
        "risk_matrix": {},
        "final_report": "",
    }

    graph = build_evaluator_graph()
    final_state = graph.invoke(initial_state)

    return final_state["final_report"]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_STRATEGY = """
    We are launching a B2B SaaS platform that helps content marketing teams 
    generate targeted content using AI. Our target customers are marketing 
    managers at SMEs with 10-200 employees. Pricing is £99/month per seat. 
    We plan to acquire customers through LinkedIn outbound and content marketing. 
    We have a working MVP, 3 beta customers, and 6 months of runway.
    """

    print("\n" + "=" * 65)
    print("  AMPLIFLOW STRATEGY EVALUATOR")
    print("  Multi-Agent Business Strategy Stress Test")
    print("=" * 65)
    print(f"\nStrategy submitted for evaluation...")
    print(f"Running 2 debate rounds + synthesis\n")

    report = evaluate_strategy(TEST_STRATEGY, rounds=2)

    print("\n" + "=" * 65)
    print("  FINAL STRATEGY REPORT")
    print("=" * 65)
    print(report)
    print("\n" + "=" * 65)