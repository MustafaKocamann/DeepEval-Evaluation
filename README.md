# 🧪 DeepEval Comprehensive Evaluation Suite: LLM, RAG & AI Agent

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/managed%20by-uv-orange.svg)](https://github.com/astral-sh/uv)
[![Evaluation Framework](https://img.shields.io/badge/framework-DeepEval-purple.svg)](https://github.com/confident-ai/deepeval)
[![LLM Provider](https://img.shields.io/badge/provider-Groq%20Cloud-f55036.svg)](https://groq.com/)
[![Agent Orchestration](https://img.shields.io/badge/agent-LangChain%20%2F%20LangGraph-green.svg)](https://github.com/langchain-ai/langchain)

A production-grade, end-to-end evaluation laboratory demonstrating testing, benchmarking, and quality assurance across the entire Generative AI lifecycle: **LLM-as-a-Judge**, **Custom DAG Pipelines**, **Retrieval-Augmented Generation (RAG)**, and **Multi-Step Tool-Using AI Agents**.

Powered by **[DeepEval](https://github.com/confident-ai/deepeval)** and ultra-fast inference via **[Groq](https://groq.com/)** (`qwen/qwen3.8-27b` and `openai/gpt-oss-120b`).

---

## 📑 Table of Contents

- [Architectural Overview](#-architectural-overview)
- [Key Evaluation Pillars](#-key-evaluation-pillars)
  - [1. LLM-as-a-Judge & G-Eval (Criteria, Steps, Rubrics)](#1-llm-as-a-judge--g-eval)
  - [2. Deterministic & Directed Acyclic Graph (DAG) Metrics](#2-deterministic--directed-acyclic-graph-dag-metrics)
  - [3. RAG Pipeline Evaluation (The RAG Triad)](#3-rag-pipeline-evaluation-the-rag-triad)
  - [4. Agentic AI & Tool Calling Evaluation (Deep Tracing)](#4-agentic-ai--tool-calling-evaluation-deep-tracing)
- [Evaluation Metric Matrix](#-evaluation-metric-matrix)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation with uv](#installation-with-uv)
  - [Environment Configuration](#environment-configuration)
- [Running the Evaluations](#-running-the-evaluations)
- [Dashboard & Telemetry (`deepeval view`)](#-dashboard--telemetry-deepeval-view)
- [Windows System Notice](#-windows-system-notice)
- [License](#-license)

---

## 🏛 Architectural Overview

```
                                      ┌────────────────────────────────────────────────────────┐
                                      │                    DeepEval Runner                     │
                                      │  (AsyncConfig, ErrorConfig, DisplayConfig, Caching)    │
                                      └─────────────────────────┬──────────────────────────────┘
                                                                │
                     ┌──────────────────────────┬───────────────┴──────────────┬──────────────────────────┐
                     ▼                          ▼                              ▼                          ▼
           ┌──────────────────┐       ┌──────────────────┐           ┌──────────────────┐       ┌──────────────────┐
           │   G-Eval Suite   │       │    DAG Metric    │           │    RAG Triad     │       │   Agent Spans    │
           ├──────────────────┤       ├──────────────────┤           ├──────────────────┤       ├──────────────────┤
           │ • Criteria Eval  │       │ • TaskNode       │           │ • Faithfulness   │       │ • ToolCorrectness│
           │ • Step-by-Step   │       │ • JudgementNode  │           │ • AnswerRelevancy│       │ • ArgumentCorrect│
           │ • Rubric Scoring │       │ • VerdictNode    │           │ • ContextPrecision│      │ • StepEfficiency │
           └────────┬─────────┘       └────────┬─────────┘           │ • ContextRecall  │       │ • PlanAdherence  │
                    │                          │                     │ • ContextRelevancy│      │ • TaskCompletion │
                    │                          │                     └────────┬─────────┘       └────────┬─────────┘
                    └──────────────────────────┴───────────────┬──────────────┴──────────────────────────┘
                                                               │
                                                               ▼
                                              ┌─────────────────────────────────┐
                                              │    Groq High-Speed Inference    │
                                              │  • Judge: qwen/qwen3.8-27b      │
                                              │  • Agent: openai/gpt-oss-120b   │
                                              └─────────────────────────────────┘
```

---

## 🔍 Key Evaluation Pillars

### 1. LLM-as-a-Judge & G-Eval
- **Custom LocalModel Integration:** Binds Groq's low-latency OpenAI-compatible endpoint as an evaluation judge (`qwen/qwen3.8-27b`).
- **Criteria-Based Assessment:** Evaluates factual consistency between generated responses and reference truth.
- **Step-Based Guidelines:** Sequential rule execution (e.g., verifying polite tone, penalizing sarcasm, checking professional etiquette).
- **Multi-Tier Rubrics:** Discrete grading thresholds mapping continuous model assessment into explicit numerical tiers (0-2: Contradictory, 3-5: Partially correct, 6-8: Minor omissions, 9-10: Fully correct).

### 2. Deterministic & Directed Acyclic Graph (DAG) Metrics
- Evaluates multi-step unstructured text pipelines (e.g., meeting transcript summarization).
- **`TaskNode`**: Extracts intermediate artifacts (e.g., summary headings).
- **`BinaryJudgementNode`**: Validates deterministic conditions (e.g., verifying if `'Intro' => 'Body' => 'Conclusion'` follows strict ordering).
- **`VerdictNode`**: Generates definitive terminal pass/fail decisions with traceable audit logs.

### 3. RAG Pipeline Evaluation (The RAG Triad)
A complete benchmark assessing both **retrieval accuracy** and **generation fidelity**:
- **Knowledge Corpus & Retrieval:** Multi-document chunking and simulated semantic lookup.
- **Generation:** Context-grounded synthesis with constrained system prompts.
- **Metrics Covered:**
  - `FaithfulnessMetric`: Measures whether output makes claims not supported by the retrieved context.
  - `AnswerRelevancyMetric`: Measures whether response directly addresses the user query.
  - `ContextualPrecisionMetric`: Checks if the most relevant chunks are ranked at the top.
  - `ContextualRecallMetric`: Validates whether all ground-truth facts were retrieved.
  - `ContextualRelevancyMetric`: Penalizes noisy or irrelevant retrieved chunks.

### 4. Agentic AI & Tool Calling Evaluation (Deep Tracing)
- **Agent Architecture:** LangChain agent powered by `ChatGroq` (`openai/gpt-oss-120b`) with tool sets:
  - `search_docs`: Documentation lookup.
  - `run_code`: Sandboxed Python execution.
  - `escalate_to_human`: Human escalation ticket generation.
- **Deep Observability (`deepeval.tracing`):** 
  - Captures nested spans, root execution graphs, and tool invocation metadata (`@observe`, `trace()`, `trace_manager`).
  - Seamlessly transforms traces into `LLMTestCase` instances containing `tools_called` and `expected_tools`.
- **Agent Metrics:**
  - `ToolCorrectnessMetric`: Did the agent invoke the right tools in the required sequence?
  - `ArgumentCorrectnessMetric`: Were the parameters passed to the tools correct and optimal?
  - `StepEfficiencyMetric`: Did the agent take redundant or circular actions?
  - `PlanAdherenceMetric` & `PlanQualityMetric`: Quality and adherence to multi-step reasoning plans.
  - `TaskCompletionMetric`: Did the overall interaction fulfill the user's intent?

---

## 📊 Evaluation Metric Matrix

| Metric Category | Metric Class | Input Parameters Evaluated | Type |
|---|---|---|---|
| **Quality & Tone** | `GEval` (Criteria) | `input`, `actual_output`, `expected_output` | LLM Judge |
| **Quality & Tone** | `GEval` (Steps) | `input`, `actual_output` | LLM Judge |
| **Quality & Tone** | `GEval` (Rubric) | `input`, `actual_output`, `expected_output` | LLM Judge (Rubric) |
| **Structural** | `DAGMetric` | `actual_output` | Graph (Deterministic + LLM) |
| **RAG** | `FaithfulnessMetric` | `actual_output`, `retrieval_context` | LLM Judge |
| **RAG** | `AnswerRelevancyMetric` | `input`, `actual_output` | LLM Judge |
| **RAG** | `ContextualPrecisionMetric` | `input`, `expected_output`, `retrieval_context` | LLM Judge |
| **RAG** | `ContextualRecallMetric` | `expected_output`, `retrieval_context` | LLM Judge |
| **RAG** | `ContextualRelevancyMetric` | `input`, `retrieval_context` | LLM Judge |
| **AI Agent** | `ToolCorrectnessMetric` | `tools_called`, `expected_tools` | LLM Judge / Exact |
| **AI Agent** | `ArgumentCorrectnessMetric` | `tools_called`, `expected_tools` | LLM Judge |
| **AI Agent** | `StepEfficiencyMetric` | `tools_called`, `actual_output` | LLM Judge |
| **AI Agent** | `PlanAdherenceMetric` | `input`, `tools_called`, `actual_output` | LLM Judge |
| **AI Agent** | `PlanQualityMetric` | `input`, `tools_called` | LLM Judge |
| **AI Agent** | `TaskCompletionMetric` | `input`, `actual_output` | LLM Judge |

---

## 📁 Repository Structure

```
DeepEval Book/
├── note.ipynb             # Interactive Jupyter Notebook containing all 4 evaluation modules
├── main.py                # Project entrypoint
├── pyproject.toml         # Dependency definitions (managed via uv)
├── uv.lock                # Deterministic lockfile
├── .env                   # Environment secrets (GROQ_API_KEY)
├── .gitignore             # Git ignore configuration
└── README.md              # Technical documentation & guide
```

---

## 🚀 Getting Started

### Prerequisites

- Python `>= 3.13`
- [uv](https://docs.astral.sh/uv/) package manager installed:
  ```bash
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- A valid [Groq Cloud API Key](https://console.groq.com/).

### Installation with uv

Clone the repository and sync the virtual environment:

```bash
git clone https://github.com/MustafaKocamann/DeepEval-Evaluation.git
cd DeepEval-Evaluation

# Install all locked dependencies into .venv
uv sync
```

### Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY="gsk_your_groq_api_key_here"
```

---

## 💻 Running the Evaluations

Open [`note.ipynb`](file:///note.ipynb) in VS Code, Cursor, or JupyterLab:

1. Select the `.venv` kernel (`.venv/Scripts/python.exe`).
2. Run cells sequentially:
   - **Cell 1–8:** Base setup, Judge initialization, and G-Eval experiments.
   - **Cell 9–16:** DAG-based custom structural evaluation.
   - **Cell 17–28:** RAG pipeline synthesis and RAG Triad benchmarking.
   - **Cell 29–40:** Agent execution, tracing, and multi-metric agent evaluation.

### Evaluation Configurations Used

The evaluations leverage fine-grained asynchronous and throttling controls to prevent rate-limiting while maintaining fast execution:

```python
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig, CacheConfig

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(max_concurrent=2, throttle_value=1.0),
    display_config=DisplayConfig(print_results=True),
    error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
    cache_config=CacheConfig(write_cache=True, use_cache=False),
)
```

---

## 📈 Dashboard & Telemetry (`deepeval view`)

DeepEval features an integrated terminal dashboard and web UI for inspecting evaluation histories, score distributions, and trace waterfall charts:

```bash
uv run deepeval view
```

You can inspect:
- Metric pass/fail rates and score breakdowns.
- Step-by-step reasoning generated by the Judge model.
- Nested agent tool traces and span execution latencies.

---

## 🪟 Windows System Notice

On Windows platforms, DeepEval's test-run cache manager relies on shared read locks (`LOCK_SH`). The standard C-runtime (`msvcrt`) does not natively provide shared locking. 

This repository includes `portalocker[win32]` (`pywin32`) in [`pyproject.toml`](file:///pyproject.toml) to ensure native Windows locking functions seamlessly:

```toml
dependencies = [
    "deepeval>=4.2.8",
    "portalocker[win32]>=4.4.0",
    ...
]
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
