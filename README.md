\# ⚡ AmpliflowMemoryEngine



A working prototype built demonstrating two core AI capabilities:



1\. \*\*Sticky Memory Model\*\* — personalised context loading for LLM-powered content generation

2\. \*\*Multi-Agent Strategy Evaluator\*\* — LangGraph debate system for business strategy stress-testing



\---



\## The Problem This Solves



LLMs have no memory between sessions. Every time a user generates content, the model starts from scratch — no knowledge of who their customers are, how their brand sounds, or what compliance rules apply.



\*\*AmpliflowMemoryEngine\*\* solves this by storing user identity in structured Markdown files and intelligently loading the right context on every request — without blowing the context window.



\---



\## Problem 1 — Sticky Memory Model



\### How It Works



```

User logs in

&#x20;    ↓

Load session state from disk (last active, last ICP, last query)

&#x20;    ↓

Build or load vector store from ICP profile Markdown files

&#x20;    ↓

User submits a content generation request

&#x20;    ↓

Semantic search → retrieve 3 most relevant ICP chunks

&#x20;    ↓

Load brand voice (always, in full)

&#x20;    ↓

Load selective compliance rules (matched to ICP type)

&#x20;    ↓

Check token budget (tiktoken cl100k\_base)

&#x20;    ↓

Assemble and return context payload → ready for any LLM

&#x20;    ↓

Write session state back to disk

```



\### Memory File Structure



```

user\_memory/

└── user\_102/

&#x20;   ├── profiles/

&#x20;   │   ├── icp\_local\_service\_business.md

&#x20;   │   ├── icp\_ecommerce\_retail\_brand.md

&#x20;   │   ├── icp\_nonprofit\_charity.md

&#x20;   │   ├── icp\_tech\_ai\_founder.md

&#x20;   │   ├── icp\_community\_culture\_org.md

&#x20;   │   ├── icp\_education\_training.md

&#x20;   │   ├── icp\_health\_wellness.md

&#x20;   │   └── icp\_industrial\_manufacturing.md

&#x20;   ├── compliance/

&#x20;   │   ├── brand\_voice.md

&#x20;   │   └── compliance\_rules.md

&#x20;   └── telemetry/

&#x20;       └── session\_state.json

```



\### Key Design Decisions



| Decision | Reason |

|---|---|

| Markdown files as memory backend | Human readable, version controllable, chunk naturally by headers |

| `MarkdownHeaderTextSplitter` | Preserves semantic relationships between headers and content |

| ChromaDB persisted to disk | No external database needed, loads instantly on repeat runs |

| `all-MiniLM-L6-v2` embeddings | Free, local, no API key, 90MB, strong semantic search quality |

| Brand voice always fully loaded | Small file, critical to every generation, never skip it |

| Compliance rules loaded selectively | Saves tokens — only load constraints relevant to the active ICP |

| k=3 retrieval | Single chunk often returns a peripheral section; 3 gives a richer ICP picture |

| `tiktoken cl100k\_base` for token counting | Same tokeniser OpenAI uses — accurate counts before hitting any API |



\### ICP Routing Accuracy



Tested across 4 query types — 4/4 correct ICP matches:



| Query | Expected ICP | Result |

|---|---|---|

| Neighbourhood Facebook post for local trades business | Local Service Business | ✓ |

| Product Hunt launch for SaaS startup targeting developers | Tech / AI Founder | ✓ |

| Donor thank-you email for charity fundraising campaign | Nonprofit / Charity | ✓ |

| Press release for factory ISO certification | Industrial / Manufacturing | ✓ |



\---



\## Problem 2 — Multi-Agent Strategy Evaluator



\### How It Works



```

User submits strategy in plain English

&#x20;         ↓

&#x20;   ┌─────────────────┐

&#x20;   │  Growth Champion │  Finds TAM, viral loops, scaling vectors

&#x20;   └────────┬────────┘

&#x20;            ↓

&#x20;   ┌─────────────────┐

&#x20;   │  Risk Challenger │  Finds burn rate issues, churn risk, weak moat

&#x20;   └────────┬────────┘

&#x20;            ↓

&#x20;   ┌─────────────────┐

&#x20;   │  ICP Simulator  │  Stress-tests as the target B2B buyer

&#x20;   └────────┬────────┘

&#x20;            ↓

&#x20;     Round < max? ──→ loop back to Growth Champion

&#x20;            ↓

&#x20;   ┌──────────────────────┐

&#x20;   │  Executive Synthesizer│  Reads full transcript → structured report

&#x20;   └──────────────────────┘

&#x20;            ↓

&#x20;   SWOT · Risk Index · GO/NO-GO · Top 3 Action Points

```



\### Built With



\- \*\*LangGraph\*\* — native state management and loop control for multi-agent workflows

\- \*\*Groq API\*\* (`llama-3.1-8b-instant`) — fast inference, free tier available

\- Configurable debate rounds (1–3) before synthesis



\---



\## Streamlit UI



Two-tab interface built for live demonstration:



\*\*Tab 1 — Memory Engine\*\*

\- Load user memory from sidebar (cached — loads once per session)

\- Submit a content generation request in plain English

\- See ICP matched, token budget breakdown, and all three context components side by side

\- Assembled system prompt ready to pass to any LLM



\*\*Tab 2 — Strategy Evaluator\*\*

\- Submit a business strategy in plain English

\- Watch each agent speak live as the debate streams

\- Full debate transcript collapsible after report renders

\- Download final report as Markdown



\---



\## Setup



\### Requirements



\- Python 3.11+

\- Groq API key (free at \[console.groq.com](https://console.groq.com))



\### Install



```bash

git clone https://github.com/devasadhu/AmpliflowMemoryEngine.git

cd AmpliflowMemoryEngine

python -m venv venv

venv\\Scripts\\activate        # Windows

pip install -r requirements.txt

```



\### Run



```bash

set GROQ\_API\_KEY=your\_key\_here     # Windows CMD

streamlit run app.py

```



On first run, the vector store builds from the ICP profile Markdown files (\~5 seconds).  

On repeat runs, it loads from disk instantly.



\### Run demo only (no UI)



```bash

python demo.py              # ICP routing accuracy test

python strategy\_evaluator.py   # Single strategy evaluation

```



\---



\## Stack



| Component | Technology |

|---|---|

| Vector store | ChromaDB (persisted to disk) |

| Embeddings | `all-MiniLM-L6-v2` via HuggingFace (local, free) |

| Text splitting | LangChain `MarkdownHeaderTextSplitter` |

| Token counting | tiktoken `cl100k\_base` |

| Multi-agent graph | LangGraph |

| LLM inference | Groq API (`llama-3.1-8b-instant`) |

| UI | Streamlit |



\---



\## Project Structure



```

AmpliflowMemoryEngine/

├── app.py                  # Streamlit UI

├── memory\_engine.py        # Core memory engine class

├── strategy\_evaluator.py   # Multi-agent LangGraph evaluator

├── demo.py                 # ICP routing demo script

├── requirements.txt

├── user\_memory/

│   └── user\_102/

│       ├── profiles/       # ICP Markdown files

│       ├── compliance/     # Brand voice + compliance rules

│       ├── telemetry/      # Session state JSON

│       └── chroma\_db/      # Persisted vector store

└── README.md

```



\---



\*Built as a prototype — June 2026\*

```

