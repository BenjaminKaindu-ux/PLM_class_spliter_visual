---
title: PLM Factory — Perceptual Learning Drills
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Adaptive perceptual learning drills
---

# 🧠 PLM Factory

A multi-agent system (built on [smolagents](https://huggingface.co/docs/smolagents)) that
generates **Perceptual Learning Modules** — rapid, timed, high-repetition classification
drills that build fast visual/pattern intuition — for three domains:

| Domain | Perceptual skill trained | Item engine (ground-truth oracle) |
|---|---|---|
| Geography | Answer geography questions, identify spatial patterns, recognize features | GeoGPT-QA + geochain datasets |
| Python | Predict code output, choose data structures, analyze complexity | CodeExercise-Python-27k dataset |
| Chess | Identify tactical patterns, find best moves, apply endgame techniques | Lichess/chess-puzzles dataset |

**This Space hosts two tabs:**
1. **GEOGRAPHY Student Demo** — the *GeoSense* drill: timed classification of
   geography questions with ARTS adaptive sequencing (accuracy **and** response time;
   categories retire at 4 consecutive fast-and-correct answers), per Kellman, Massey & Son
   (2010), *Topics in Cognitive Science*.
2. **Agent Debug (dev)** — pick a domain + concept and watch a real `smolagents` `CodeAgent`
   (Qwen2.5-Coder-32B-Instruct) generate a live item end to end, with its code trace shown so
   you can verify it calls the domain's sanctioned generator tool rather than asserting an
   answer itself. Deliberately built as fixed dropdowns rather than smolagents' built-in
   free-text `GradioUI` chat, since an open-ended chat into a code-executing agent is a real
   prompt-injection surface on a public Space.

## Architecture (v2)

Three agents: **Retrieval Agent** (Chroma ×3 collections, `BAAI/bge-m3` embeddings via the HF
Inference API) + **Research Agent** (DuckDuckGo web search → cached, per-topic concept briefs;
`Qwen2.5-Coder-32B-Instruct`) + a single **PLM Agent** (CodeAgent, domain as a parameter;
`Qwen2.5-Coder-32B-Instruct`) whose chain-of-thought only picks *category / difficulty /
instance parameters* — all item content and answer keys come from deterministic dataset oracles,
never LLM assertion, enforced live via a `step_callbacks` guard. Full design: `ARCHITECTURE.md`.

## Repo layout

```
app.py                        # this Space: student demo tab + agent debug tab
src/config.py                 # model assignments, Chroma paths, ARTS constants
src/plm_core/arts.py          # ARTS adaptive sequencing tracker (deterministic)
src/plm_core/plm_agent.py     # PLM Agent (CodeAgent, domain as a parameter)
src/plm_core/retrieval.py     # Retrieval Agent + retrieve_course_context tool
src/plm_core/research.py      # Research Agent (web -> cached per-topic briefs)
src/plm_core/embeddings.py    # bge-m3 embeddings via HF Inference API
src/generators/geography.py   # GeoGPT-QA + geochain dataset oracle items
src/generators/python.py      # CodeExercise-Python-27k dataset oracle items
src/generators/chess.py       # Lichess/chess-puzzles dataset oracle items
data/chroma/                  # pre-built Chroma DB (geography, python, chess collections)
ARCHITECTURE.md               # confirmed v2 system design
PROPOSAL.md                   # superseded v1 (kept for item schema / ARTS detail)
requirements.txt              # what's actually installed on this Space
requirements-full.txt         # full project stack incl. local-only ingestion/observability
```

## Datasets

This project uses the following HuggingFace datasets:

| Dataset | Domain | Size | Format |
|---------|--------|------|--------|
| [GeoGPT-Research-Project/GeoGPT-QA](https://huggingface.co/datasets/GeoGPT-Research-Project/GeoGPT-QA) | Geography | 41.4K QA pairs | CSV/Parquet |
| [sahitiy51/geochain](https://huggingface.co/datasets/sahitiy51/geochain) | Geography | 1.4M rows | Parquet |
| [codefuse-ai/CodeExercise-Python-27k](https://huggingface.co/datasets/codefuse-ai/CodeExercise-Python-27k) | Python | 27K exercises | JSON/CSV |
| [Lichess/chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles) | Chess | 6.1M puzzles | Parquet |

## Quick Start

1. Visit the [live Space](https://huggingface.co/spaces/BenjaminKaindu0506/plm-factory)
2. Select a domain from the dropdown (GEOGRAPHY, PYTHON, or CHESS)
3. Click "Next trial ▶" to start the drill
4. Answer questions as fast as you can
5. Watch categories retire as you master them

## Development

```bash
# Clone the repo
git clone https://huggingface.co/spaces/BenjaminKaindu0506/plm-factory.git
cd plm-factory

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

## License

MIT
