# MES Framework

**MES** (Multi-Agent Evidence Synthesis) is an agentic framework for **medical evidence synthesis**. By coordinating six specialized agents (Supervisor, LiteratureMiner, RWD-Analyst, KG-Specialist, Statistician, and SafeChecker), it transforms natural-language clinical questions into structured reports integrating **published literature**, **real-world evidence**, and **biomedical knowledge graphs**.

This repository contains the implementation of the MES framework accompanying the manuscript *"MES: A Multi-Agent Evidence Synthesis System for Medical Decision-Making"*.

## 🚀 Features

* **Multi-Agent Workflow:** Structures clinical questions using PICO and TTR, coordinates specialist agents, and supports iterative refinement.
* **Multi-Source Evidence:** Retrieves biomedical literature and trial records from PubMed and ClinicalTrials.gov, alongside structured and embedding-based knowledge-graph retrieval.
* **Real-World Evidence Generation:** Constructs clinical cohorts and supports target trial emulation, prediction, descriptive analysis, clustering, and patient-similarity retrieval.
* **Statistical Synthesis & Claim Screening:** Performs covariate balancing, survival analysis, meta-analysis of compatible studies, cross-source divergence assessment, and claim-level evidence-support screening.
* **Hybrid LLM Support:** Switches between cloud-based OpenAI models and local models through **Ollama** using a shared model registry.

## 🛠️ Prerequisites & Installation

### 1. System Requirements

* **Python 3.11** and **Conda**.
* **Ollama** for local model execution.
* Access to MIMIC-IV tables and biomedical knowledge-graph resources (e.g., [iBKH](https://github.com/wcm-wanglab/iBKH)).

### 2. Clone the Repository

```bash
git clone https://github.com/TrialLab/MES.git
cd MES
```

### 3. Environment Setup

Create the environment using the provided specification:

```bash
conda env create -f environment.yml
conda activate mes
```

### 4. Setting up Local Models (Ollama)

With Ollama installed, pull a local model:

```bash
ollama pull gemma3:4b
```

Ensure the Ollama server is running at `localhost:11434`. Model aliases are defined in `llm_zoo.py`.

### 5. API & Data Configuration

Copy `.env.example` to `.env` and configure the data paths. Set `OPENAI_API_KEY` when using a cloud model:

```ini
OPENAI_API_KEY=your_key_here
MIMIC_PATH=Data/mimiciv
KG_TRIPLETS_CSV=Data/KG/triplets.csv
```

The knowledge-graph CSV uses four columns: `head`, `relation`, `tail`, and `source`.

## 🏃 Usage

The core workflow is driven by the Jupyter Notebook `MES-demo.ipynb`.

### Running an Investigation

```bash
jupyter notebook MES-demo.ipynb
```

Set the clinical question, select the LLM backend, and confirm the data paths in the configuration cell. Run the cells sequentially to inspect the evidence, analytical outputs, claim-screening results, and final synthesis.

## 🔌 External Services

MES uses PubMed and ClinicalTrials.gov for evidence retrieval, Hugging Face for the biomedical embedding model, and Ollama or the OpenAI API for language-model inference.

## 📂 Repository Structure

* `MES-demo.ipynb`: Main notebook for configuring and running investigations.
* `mes_pipeline.py`: Agent orchestration, iterative refinement, and output saving.
* `Supervisor.py`: Clinical question structuring and report assembly.
* `LiteratureMiner.py`: Literature retrieval and evidence extraction.
* `RWDAnalyst.py`, `rwd_preprocessor.py`, `rwd_extended_functions.py`: Cohort construction and real-world analysis pipelines.
* `emulation_utils.py`: Covariate balancing, causal inference, and survival analysis.
* `KGSpecialist.py`: Structured and embedding-based graph retrieval.
* `Statistician.py`: Quantitative synthesis and cross-source comparison.
* `SafeChecker.py`: Claim-support screening and consistency checks.
* `llm_zoo.py`: Cloud and local model registry.
