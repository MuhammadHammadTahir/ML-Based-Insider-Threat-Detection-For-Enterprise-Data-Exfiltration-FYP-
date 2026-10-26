<div align="center">

# ML-Based Insider Threat Detection for Enterprise Data Exfiltration

**A machine-learning and multi-agent approach to detecting anomalous user behaviour in enterprise activity logs**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/ML-Anomaly%20Detection-purple)
![ELK Stack](https://img.shields.io/badge/Observability-ELK%20Stack-yellow)
![Project](https://img.shields.io/badge/Project-Final%20Year%20Project-green)

</div>

---

## Overview

Insider threats are difficult to detect because malicious employees, compromised accounts, and negligent users may already have legitimate access to enterprise systems. Their activity can resemble normal work, allowing suspicious behaviour to go unnoticed by static rules or signature-based controls.

This project presents an **ML-based insider threat detection pipeline** that analyses activity represented in the CERT Insider Threat Dataset, aggregates raw events into user sessions, and combines multiple detection agents to identify behaviour that may indicate an insider threat or data exfiltration.

The system also demonstrates a log-monitoring and visualisation pipeline using **Filebeat, Logstash, Elasticsearch, and Kibana (ELK)**. The project is an academic research prototype, not a production-ready security product.

## Key Features

- **Behavioural anomaly detection:** Uses a Contractive Autoencoder (CAE) to learn patterns from numerical user-activity features.
- **Semantic analysis:** Uses `all-mpnet-base-v2` embeddings and `distilbert-base-uncased` to analyse textual activity and compare it with suspicious-content anchors.
- **User-specific temporal analysis:** Uses Z-scores to identify deviations from a user's historical activity patterns.
- **Multi-agent score fusion:** Combines agent outputs and interaction features through an Isolation Forest-based orchestrator.
- **Centralised log ingestion:** Demonstrates forwarding and processing logs through Filebeat and Logstash.
- **Search and storage:** Indexes structured events in Elasticsearch for querying and investigation.
- **Security dashboards:** Uses Kibana for exploring activity and reviewing detection results.
- **Session-based inference:** Aggregates events into structured user sessions before model inference.

## System Architecture

At a high level, the project follows this workflow:

```text
CERT Activity Dataset
        |
        v
Data Replay / Log Generator
(CERT records -> JSON / NDJSON events)
        |
        v
Filebeat
        |
        v
Logstash
        |
        v
Elasticsearch  <-------------------------------+
        |                                       |
        v                                       |
Session Aggregation                             |
        |                                       |
        v                                       |
Multi-Agent Detection                           |
        |                                       |
        +--> Behavioural Agent (CAE)             |
        +--> Semantic Agent (MPNet + DistilBERT) |
        +--> Temporal Agent (Z-score analysis)   |
        +--> Orchestrator (Isolation Forest)     |
        |                                       |
        v                                       |
Risk / Anomaly Results -------------------------+
        |
        v
Kibana Dashboard and Analyst Investigation
```

The precise integration and execution sequence may depend on the scripts and configuration in each project folder. The diagram describes the intended high-level architecture documented in the project presentation.

## Multi-Agent Detection Methodology

The system uses four specialised agents. Their outputs are shared through a **blackboard**, a common data structure where agents can publish and read scores and flags.

### 1. Behavioural Agent: Contractive Autoencoder

The behavioural agent learns patterns in normal numerical activity and identifies sessions that differ from those patterns.

Examples of activity categories represented in the feature set include:

- Email activity
- File activity and transfers
- USB usage
- Session-level behaviour

The presentation describes scaling 12 numerical features and flagging sessions whose reconstruction error exceeds a threshold based on `μ + 2σ`. Scores and flags are then made available to the shared blackboard.

### 2. Semantic Agent: MPNet and DistilBERT

The semantic agent analyses textual activity, such as available email, HTTP, and file-related content, to identify content that may be suspicious.

The described components are:

- **Embedding model:** `sentence-transformers/all-mpnet-base-v2`
- **Text classifier:** `distilbert-base-uncased`

MPNet embeddings are compared with malicious-content anchor texts using cosine similarity. DistilBERT classifies text snippets, and the resulting confidence or anomaly information is published to the blackboard. The presentation also describes a score adjustment when the behavioural agent has flagged a session.

### 3. Temporal Agent: User-Specific Z-Score Analysis

The temporal agent compares activity against historical behaviour to identify unusual timing or frequency patterns.

Examples include:

- Login-time patterns
- File-access frequency
- Email-sending times
- USB usage history

Where a user's history is available, the agent uses a user-specific baseline. A global baseline is described for unknown users.

### 4. Orchestrator Agent: Isolation Forest

The orchestrator collects agent scores and flags from the blackboard, constructs a combined feature vector, and makes the final risk decision.

The presentation describes a **10-dimensional feature vector** containing agent scores, interaction features, and agreement signals. An Isolation Forest trained on benign-only data assigns an anomaly score; sessions exceeding the configured threshold are classified as malicious.

> **Important:** A model flag indicates potentially anomalous activity. It is not, by itself, proof of malicious intent. Detection results should be reviewed in context.

## Data and Evaluation

### Dataset

The project uses the **CERT Insider Threat Dataset**, which contains simulated enterprise activity rather than live endpoint telemetry. The project workflow replays dataset activity as structured events for ingestion and analysis.

The project also describes extending the working dataset with textual-content columns to support semantic analysis alongside numerical behavioural features.

Use the dataset in accordance with its original terms and any applicable access restrictions. The dataset itself may need to be obtained separately and placed in the location expected by the project's scripts.

### Evaluation

The project presentation includes comparisons between individual detection agents and the multi-agent system, along with ROC-AUC and precision-recall visualisations.

Reported project evaluation figures include:

| Metric | Reported result |
|---|---:|
| Multi-agent system ROC-AUC | 0.9930 |
| Comparison/baseline ROC-AUC | 0.9271 |
| Recall | 91.49% |

These figures are reproduced as project-reported results, not as an independently reproduced benchmark. Refer to the implementation, evaluation scripts, dataset split, threshold selection, and full report for the exact experimental setup and interpretation. Results on a simulated dataset do not guarantee the same performance in a real enterprise environment.

## Deployment Pipeline

The project presentation describes the following deployment components:

| Component | Purpose |
|---|---|
| Data replay / generator | Converts CERT activity into structured JSON or NDJSON events and simulates log flow |
| Filebeat | Forwards log events from the collection environment |
| Logstash | Processes and forwards events to Elasticsearch |
| Elasticsearch | Stores and indexes structured logs for search |
| Session aggregation | Groups events into user sessions and extracts model features |
| Model inference engine | Applies the trained model or detection pipeline to aggregated sessions |
| Kibana | Supports dashboard visualisation and analyst investigation |

The presentation's lab setup describes Filebeat 8.12.0 on a Windows VM and Elasticsearch 8.12.1 on an Ubuntu VM, with Logstash receiving events on port `5044` and Kibana available on port `5601`. These are project-lab details, not universal defaults for every deployment.

### Dashboard files

The repository contains Kibana dashboard export files:

- `insider_threat_dashboard(1).ndjson`
- `insider_threat_dashboard_fixed(2).ndjson`

To try an export, open Kibana and use **Stack Management → Saved Objects → Import**, then select the `.ndjson` file. Review the target index patterns and data views after importing. Dashboard visualisations may not populate until the expected indices and fields exist in Elasticsearch. The exact menu labels can vary by Kibana version.

## Repository Structure

The following top-level items are present in the repository. Folder-level descriptions are intentionally high-level; inspect the source files in each folder to confirm individual scripts and entry points.

```text
.
├── Code/                                  # Main project code
├── Deployment_Pipeline/                   # Log ingestion / deployment components
├── Genrator/                              # Data generation or replay-related work
├── OldCode/                               # Earlier or superseded implementation
├── MAS_Testing.ipynb                      # Multi-agent system experiments
├── OpenHousePresentation.pptx              # Project presentation
├── insider_threat_dashboard(1).ndjson      # Kibana dashboard export
├── insider_threat_dashboard_fixed(2).ndjson# Alternate/fixed dashboard export
└── README.md
```

The folder name `Genrator` is preserved as it currently appears in the repository. It can be renamed to `Generator` later if desired, provided any scripts, imports, or documentation that refer to it are updated.

## Getting Started

Because execution commands, dependency files, model artefacts, and dataset paths can vary across the repository's subfolders, use this workflow to set up the project without assuming a single entry point.

### Prerequisites

- Python environment compatible with the project's scripts
- Git
- Access to the CERT Insider Threat Dataset
- Required Python packages for the relevant scripts
- Elasticsearch, Logstash, Filebeat, and Kibana if reproducing the ELK deployment
- Trained model artefacts and configuration files, if they are not generated during setup

### 1. Clone the repository

```bash
git clone https://github.com/MuhammadHammadTahir/ML-Based-Insider-Threat-Detection-For-Enterprise-Data-Exfiltration-FYP-.git
cd ML-Based-Insider-Threat-Detection-For-Enterprise-Data-Exfiltration-FYP-
```

### 2. Inspect the project folders

Start by reviewing `Code/`, `Deployment_Pipeline/`, and `Genrator/` to identify the intended entry points, dependency files, configuration files, and expected input/output paths.

```bash
# Windows PowerShell
Get-ChildItem -Recurse -File

# Linux / macOS
find . -maxdepth 4 -type f
```

### 3. Install dependencies

If a `requirements.txt` or environment file is provided for a particular component, create an isolated virtual environment and install that component's documented dependencies. Do not install guessed package versions; follow the files and instructions supplied with the relevant module.

Example virtual-environment setup:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install dependencies using the appropriate requirements file once you have identified it:

```bash
pip install -r path/to/requirements.txt
```

Replace `path/to/requirements.txt` with the actual path in the component you intend to run.

### 4. Configure data and services

1. Obtain the CERT dataset separately and confirm its expected directory and file names.
2. Review the generator or replay scripts and configure their input and output paths.
3. Configure Filebeat and Logstash to use matching input paths, ports, and output destinations.
4. Confirm that Elasticsearch is receiving the expected structured events.
5. Run the session aggregation and inference components according to their own instructions.
6. Import a dashboard export into Kibana and verify that its data views match the indices produced by your pipeline.

**Note:** This README does not invent a one-command launch process. The exact commands should be added after verifying the current entry points and configuration files in the repository.

## Limitations and Responsible Use

- This is an academic prototype evaluated using simulated CERT activity, not a validated production detection service.
- Anomaly detection can generate false positives and false negatives.
- Semantic analysis depends on the text fields available in the dataset and the way the working dataset was constructed.
- Model performance depends on preprocessing, feature construction, data splits, thresholds, and the evaluation methodology.
- Real-world deployment would require additional testing, monitoring, access controls, secure configuration, privacy safeguards, and performance evaluation.
- Do not ingest real employee communications or other sensitive records without appropriate authorisation, a lawful basis, and suitable data-protection controls.
- Do not commit credentials, secrets, private datasets, or sensitive operational logs to the repository.

## Future Improvements

Potential next steps for the research prototype include:

- Documenting reproducible training and evaluation commands
- Adding component-specific dependency files and configuration examples
- Providing a sample, non-sensitive event file for pipeline testing
- Documenting index mappings and required Kibana data views
- Adding automated tests for data transformation, session aggregation, and inference
- Evaluating false-positive rates and performance on held-out scenarios
- Improving model explainability and the analyst investigation workflow
- Documenting deployment security and privacy requirements

## Project Team

**Pakistan Institute of Engineering and Applied Sciences (PIEAS)**

- Ali Raza
- Muhammad Hammad Tahir
- Muhammad Talal Haider

**Supervisor:** Dr. Asifullah Khan

## Academic Context

This project was developed as a Final Year Project (FYP) at PIEAS. The repository includes project code, experimentation, deployment-pipeline work, dashboard exports, and presentation material.

## Disclaimer

This repository is intended for academic, research, and educational use. Detection outputs are indicators for further investigation and should not be treated as definitive evidence of malicious behaviour.

