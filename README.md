
---
title: EvalGated News Classifier
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# EvalGated News Classifier API

A production-grade, evaluation-gated news classification REST API built with FastAPI, wrapped in Gradio, and hosted on Hugging Face Spaces. The service leverages Groq's LLM endpoint (`openai/gpt-oss-120b`) for deterministic zero-shot text classification into four core AG News categories, backed by local guardrails and automated CI/CD quality gates.

---

## Table of Contents

* [Overview](https://www.google.com/search?q=%23overview)
* [Features](https://www.google.com/search?q=%23features)
* [Tech Stack](https://www.google.com/search?q=%23tech-stack)
* [Architecture](https://www.google.com/search?q=%23architecture)
* [Database Design](https://www.google.com/search?q=%23database-design)
* [API Endpoints](https://www.google.com/search?q=%23api-endpoints)
* [Installation](https://www.google.com/search?q=%23installation)
* [Environment Variables](https://www.google.com/search?q=%23environment-variables)
* [Usage](https://www.google.com/search?q=%23usage)
* [Mathematical Evaluation & Benchmarks](https://www.google.com/search?q=%23mathematical-evaluation--benchmarks)
* [Screenshots](https://www.google.com/search?q=%23screenshots)
* [Deployment](https://www.google.com/search?q=%23deployment)
* [Future Improvements](https://www.google.com/search?q=%23future-improvements)
* [Credits](https://www.google.com/search?q=%23credits)
* [License](https://www.google.com/search?q=%23license)

---

## Overview

This repository implements an Evaluation-Gated News Classifier. The system enforces strict quality standards by gating deployments through automated evaluation pipelines. If a code change or prompt update causes the Macro F1 score on a golden dataset to drop below `0.75`, the GitHub Actions CI/CD pipeline fails and blocks deployment to Hugging Face Spaces.

## Features

* **Deterministic Classification:** Enforces `temperature=0.0` for consistent, reproducible category predictions.
* **Local Guardrail Intercepts:** Detects empty inputs and prompt injection attempts locally before making external LLM calls, protecting token budgets.
* **CI/CD Quality Gating:** Automated GitHub Actions workflows run offline evaluations and enforce metric thresholds prior to deployment.
* **Dual Interface:** Provides both programmatic FastAPI REST endpoints and an interactive Gradio web interface.
* **Production Load Tested:** Verified 0.00% error rate across 50 consecutive benchmark requests against live production infrastructure.

## Tech Stack

* **Frameworks:** FastAPI, Gradio, Pydantic
* **LLM Engine & API:** OpenAI Python SDK, Groq API (`openai/gpt-oss-120b`)
* **Evaluation & Analytics:** scikit-learn, NumPy
* **CI/CD & Hosting:** GitHub Actions, Hugging Face Spaces (ZeroGPU / CPU Basic)

## Architecture

Client requests enter through either the Gradio user interface or programmatic REST API endpoints (`/classify_text` and `/classify`). Payload inputs undergo local pre-validation sanitization before being forwarded via HTTPS to the external Groq LLM API.

```
[Client / User]
       │
       ▼
[FastAPI / Gradio Middleware] ───(Local Check)───► Prompt Injection Intercept ("flag_for_human")
       │
       ▼ (HTTPS)
[Groq API: gpt-oss-120b]
       │
       ▼
[Response Validation] ───► [Return Predicted Label]

```

## Database Design

This application operates statelessly and does not require an active SQL/NoSQL database connection. Evaluation datasets and benchmark schemas are managed via structured JSON files:

* **`golden_set.json`:** Stores the 105-item evaluation dataset containing benchmark AG News samples and adversarial edge cases.
```json
[
  {
    "text": "Short news text sample...",
    "expected_label": "Business"
  }
]

```



## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check endpoint returning operational status. |
| `POST` | `/classify` | Primary classification endpoint accepting Pydantic JSON body. |
| `GET/POST` | `/classify_text` | Gradio interface binding route. |

### Sample Request (`POST /classify`)

```json
{
  "text": "Global oil prices surge after major trade route faces unexpected maritime blockage."
}

```

### Sample Response

```json
{
  "label": "Business"
}

```

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/vivekbiju/p8-EvalGated-News-Classifier.git
cd p8-EvalGated-News-Classifier

```


2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



## Environment Variables

Create a `.env` file in the root directory and configure the following key:

```env
GROQ_API_KEY=your_groq_api_key_here

```

For GitHub Actions, add `GROQ_API_KEY`, `HF_TOKEN`, and `HF_USERNAME` under **Settings → Secrets and variables → Actions**.

## Usage

### Running Locally

To launch the FastAPI server locally:

```bash
uvicorn main:app --reload --port 7860

```

To run the standalone evaluation suite:

```bash
python eval.py

```

To execute live load testing (50 requests):

```bash
python load_test.py

```

## Mathematical Evaluation & Benchmarks

The system is evaluated on a 105-item golden set using standard multi-class evaluation metrics.

### Key Equations

* **Precision ($P$), Recall ($R$), and F1 Score ($F_1$):**

$$P = \frac{TP}{TP + FP}, \quad R = \frac{TP}{TP + FN}, \quad F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$


* **Macro Average F1 Score ($F_{1,\text{macro}}$):**
Calculated across the $N=4$ core categories ($C = \{\text{Business, Sci/Tech, Sports, World}\}$):

$$F_{1,\text{macro}} = \frac{1}{N} \sum_{c \in C} F_{1,c}$$


* **Cohen's Kappa ($\kappa$):**

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$



### Production Benchmark Results

| Metric | Target / Production Value |
| --- | --- |
| **Macro F1 Score** | **0.8909** (Threshold: `0.75`) |
| **Cohen's Kappa ($\kappa$)** | **0.8475** |
| **Mean Latency** | `2.0243s` |
| **p95 Latency** | `3.0241s` |
| **Load Test Error Rate** | **0.00%** (50/50 successful calls) |

### Prompt Engineering Delta

* **Production Specific Prompt Macro F1:** `0.8909`
* **Vague Prompt (Appendix) Macro F1:** `0.6994`
* **Quantitative Delta:** **+0.1915 F1** (+19.15% accuracy gain)

## Screenshots

* **Passing CI/CD Build:** Successful evaluation gate and deployment sync in GitHub Actions.
* **Failing CI/CD Build:** Failure gate execution (`sys.exit(1)`) triggered when Macro F1 fell below the `0.75` threshold.

## Deployment

Deployment is fully automated using GitHub Actions (`.github/workflows/eval.yml`). Upon a `git push origin main`:

1. GitHub Actions sets up Python 3.11 and installs dependencies.
2. `eval.py` executes against the Groq API.
3. If Macro F1 $\ge 0.75$, the pipeline syncs the codebase to Hugging Face Spaces.

## Future Improvements

* Implement exponential backoff retry mechanisms for upstream HTTP 429 rate limit handling.
* Expand the golden set dataset size to include more multi-label edge cases.
* Integrate Prometheus metrics export endpoints for live latency and error monitoring.

## Credits

Developed by **Vivek Biju** as part of the Option D Evaluation-Gated AI Engineering coursework. Built using Groq API infrastructure and Hugging Face Spaces.

## License

This project is licensed under the [MIT LICENSE](./LICENSE) - see the LICENSE file for details.
