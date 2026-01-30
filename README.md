# LLM Reproducibility

Evaluates the reproducibility of large language models on oncology trial classification across different temperature settings and reasoning levels.

## Overview

This project tests whether LLM outputs are consistent across repeated runs with identical inputs. Models classify randomized controlled oncology trials as POSITIVE (met primary endpoint) or NEGATIVE (did not meet primary endpoint).

## Models Tested

- **OpenAI:** GPT-5.2 (`gpt-5.2-2025-12-11`)
- **Google:** Gemini 3 Flash (`gemini-3-flash-preview`)

## Experimental Conditions

| Parameter | Values |
|-----------|--------|
| Temperature | 0.0, 0.5, 1.0, 1.5, 2.0 |
| Reasoning Level | Minimal, Low, Medium, High |
| Repetitions | 3 per condition |

## Project Structure

```
├── run_evaluation.py      # Main evaluation pipeline
├── analyze_results.py     # Results analysis and metrics
├── visualizations.py      # Plot generation
├── data/                  # Input trial data
├── results/               # Output predictions and metrics
└── plots/                 # Generated visualizations
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation
python run_evaluation.py

# Analyze results
python analyze_results.py

# Generate plots
python visualizations.py
```

Requires a `.env` file with `OPENAI_API_KEY` and `GEMINI_API_KEY`.
