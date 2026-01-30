#!/usr/bin/env python3
"""
LLM Reproducibility Evaluation Pipeline

Evaluates OpenAI and Gemini models on oncology trial classification.
Supports resume from checkpoint and saves after each API call.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# API clients (imported conditionally)
openai_client = None
genai = None


PROMPT_TEMPLATE = """You will be provided with the title and abstract of a randomized controlled oncology trial. Your task will be to classify if the trial was positive, i.e. if it met its primary endpoint, or negative, i.e. if it did not meet its primary endpoint. Your response should be either the word POSITIVE (in all caps) or NEGATIVE (in all caps).

Title: {title}

Abstract: {abstract}"""


def init_openai():
    """Initialize OpenAI client."""
    global openai_client
    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI()
    return openai_client


def init_gemini():
    """Initialize Gemini client using the new google-genai SDK."""
    global genai
    if genai is None:
        from google import genai as genai_module
        genai = genai_module.Client(api_key=os.environ["GEMINI_API_KEY"])
    return genai


def build_prompt(title: str, abstract: str) -> str:
    """Build the classification prompt."""
    return PROMPT_TEMPLATE.format(title=title, abstract=abstract)


def parse_output(raw_output: str | None) -> str:
    """Parse model output to POSITIVE, NEGATIVE, INVALID, or ERROR."""
    if raw_output is None:
        return "ERROR"
    text = raw_output.strip().upper()
    if text == "POSITIVE":
        return "POSITIVE"
    elif text == "NEGATIVE":
        return "NEGATIVE"
    else:
        return "INVALID"


def call_openai(model: str, prompt: str, temperature: float | None, reasoning_level: str) -> str | None:
    """Call OpenAI API. Returns raw output or None on error."""
    try:
        client = init_openai()

        if reasoning_level == "none":
            # Temperature sweep without reasoning
            # Only temperature influences output variability
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=500  # High cap to capture hallucinations
            )
        else:
            # Reasoning modes (temperature not allowed)
            # Only reasoning_effort influences output variability
            # max_completion_tokens includes reasoning tokens
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=reasoning_level,
                max_completion_tokens=32000  # High cap for reasoning + output
            )

        return response.choices[0].message.content.strip() if response.choices[0].message.content else None
    except Exception as e:
        print(f"  OpenAI API error: {e}", file=sys.stderr)
        return None


def call_gemini(model: str, prompt: str, temperature: float, thinking_level: str) -> str | None:
    """Call Gemini API using the new google-genai SDK. Returns raw output or None on error."""
    try:
        from google.genai import types

        client = init_gemini()

        # Temperature and thinking_level are the only parameters influencing variability
        # max_output_tokens is separate from thinking tokens when using thinking_level
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=500,  # High cap to capture hallucinations
                thinking_config=types.ThinkingConfig(
                    thinking_level=thinking_level
                )
            )
        )

        return response.text.strip() if response.text else None
    except Exception as e:
        print(f"  Gemini API error: {e}", file=sys.stderr)
        return None


def call_api(vendor: str, model: str, temperature: float | None, reasoning_level: str, prompt: str) -> str | None:
    """Dispatch to appropriate API."""
    if vendor == "openai":
        return call_openai(model, prompt, temperature, reasoning_level)
    elif vendor == "gemini":
        return call_gemini(model, prompt, temperature, reasoning_level)
    else:
        raise ValueError(f"Unknown vendor: {vendor}")


def generate_settings() -> list[tuple[str, str, float | None, str]]:
    """Generate all (vendor, model, temperature, reasoning_level) settings."""
    settings = []

    # OpenAI: reasoning=none with temperature sweep
    for temp in [0, 0.5, 1, 1.5, 2]:
        settings.append(("openai", "gpt-5.2-2025-12-11", temp, "none"))

    # OpenAI: reasoning levels without temperature
    for reason in ["low", "medium", "high", "xhigh"]:
        settings.append(("openai", "gpt-5.2-2025-12-11", None, reason))

    # Gemini: all combinations
    for temp in [0, 0.5, 1, 1.5, 2]:
        for think in ["MINIMAL", "LOW", "MEDIUM", "HIGH"]:
            settings.append(("gemini", "gemini-3-flash-preview", temp, think))

    return settings


def load_existing_predictions(predictions_path: Path) -> set[tuple]:
    """Load existing predictions and return set of completed keys."""
    completed = set()
    if predictions_path.exists():
        df = pd.read_csv(predictions_path)
        for _, row in df.iterrows():
            key = (
                row['trial_idx'],
                row['vendor'],
                row['model'],
                row['temperature'] if pd.notna(row['temperature']) else None,
                row['reasoning_level'],
                row['replicate']
            )
            completed.add(key)
    return completed


def save_prediction(predictions_path: Path, trial_idx: int, vendor: str, model: str,
                   temperature: float | None, reasoning_level: str, replicate: int,
                   raw_output: str | None, parsed_output: str, ground_truth: str):
    """Append a single prediction to CSV."""
    row = {
        'trial_idx': trial_idx,
        'vendor': vendor,
        'model': model,
        'temperature': temperature,
        'reasoning_level': reasoning_level,
        'replicate': replicate,
        'raw_output': raw_output,
        'parsed_output': parsed_output,
        'ground_truth': ground_truth
    }

    df = pd.DataFrame([row])

    # Append mode: write header only if file doesn't exist
    write_header = not predictions_path.exists()
    df.to_csv(predictions_path, mode='a', header=write_header, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run LLM reproducibility evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Print settings count without API calls")
    parser.add_argument("--max-trials", type=int, default=None, help="Limit number of trials to process")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Paths
    script_dir = Path(__file__).parent
    trials_path = script_dir / "data" / "trials.csv"
    predictions_path = script_dir / "results" / "predictions.csv"

    # Ensure results directory exists
    predictions_path.parent.mkdir(exist_ok=True)

    # Load trials
    trials_df = pd.read_csv(trials_path)
    if args.max_trials:
        trials_df = trials_df.head(args.max_trials)

    # Generate settings
    settings = generate_settings()

    # Dry run mode
    if args.dry_run:
        total_calls = len(trials_df) * len(settings) * 3  # 3 replicates
        print(f"Settings: {len(settings)}")
        print(f"  OpenAI settings: 9 (5 temp + 4 reasoning)")
        print(f"  Gemini settings: 20 (5 temp x 4 thinking)")
        print(f"Trials: {len(trials_df)}")
        print(f"Replicates: 3")
        print(f"Total API calls: {total_calls}")
        return

    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set", file=sys.stderr)
    if not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set", file=sys.stderr)

    # Load existing predictions for resume
    completed = load_existing_predictions(predictions_path)
    print(f"Loaded {len(completed)} existing predictions")

    # Calculate totals
    total_calls = len(trials_df) * len(settings) * 3
    remaining_calls = total_calls - len(completed)
    print(f"Total calls needed: {total_calls}, remaining: {remaining_calls}")

    # Main evaluation loop
    # Process replicates in separate passes to avoid consecutive identical calls
    # Pass 1: all trials × all settings × rep=1
    # Pass 2: all trials × all settings × rep=2
    # Pass 3: all trials × all settings × rep=3
    call_count = 0
    for replicate in [1, 2, 3]:
        print(f"\n=== Replicate {replicate} of 3 ===\n")
        for trial_idx, row in trials_df.iterrows():
            title = row['title']
            abstract = row['abstract']
            ground_truth = row['Annotation_accept']

            for vendor, model, temp, level in settings:
                # Check if already done
                key = (trial_idx, vendor, model, temp, level, replicate)
                if key in completed:
                    continue

                # Build prompt and call API
                prompt = build_prompt(title, abstract)
                print(f"[{call_count + 1}/{remaining_calls}] Trial {trial_idx}, {vendor}, temp={temp}, level={level}, rep={replicate}")

                raw_output = call_api(vendor, model, temp, level, prompt)
                parsed_output = parse_output(raw_output)

                # Save immediately
                save_prediction(
                    predictions_path, trial_idx, vendor, model, temp, level,
                    replicate, raw_output, parsed_output, ground_truth
                )

                call_count += 1

                # Small delay between calls to avoid rate limiting
                time.sleep(0.5)

                # Print progress
                if parsed_output in ("INVALID", "ERROR"):
                    print(f"  -> {parsed_output}: {raw_output}")
                else:
                    print(f"  -> {parsed_output}")

    print(f"\nCompleted {call_count} API calls")
    print(f"Results saved to {predictions_path}")


if __name__ == "__main__":
    main()
