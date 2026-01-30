#!/usr/bin/env python3
"""
Analyze LLM Reproducibility Evaluation Results

Computes:
- Fleiss' kappa per setting (agreement across 3 replicates)
- Majority vote predictions
- F1 scores per setting
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters


def compute_majority_vote(predictions: list[str]) -> str:
    """
    Compute majority vote from list of predictions.
    Returns POSITIVE, NEGATIVE, or INVALID for ties/conflicts.
    """
    counter = Counter(predictions)
    most_common = counter.most_common()

    if len(most_common) == 1:
        # All same
        return most_common[0][0]
    elif most_common[0][1] > most_common[1][1]:
        # Clear majority
        return most_common[0][0]
    else:
        # Tie - mark as invalid
        return "INVALID"


def compute_fleiss_kappa_for_setting(df_setting: pd.DataFrame) -> float:
    """
    Compute Fleiss' kappa for a setting across all trials.

    Args:
        df_setting: DataFrame with columns [trial_idx, replicate, parsed_output]

    Returns:
        Fleiss' kappa value
    """
    # Map outputs to categories
    category_map = {"POSITIVE": 0, "NEGATIVE": 1, "INVALID": 2, "ERROR": 2}

    # Build matrix: rows = trials, columns = raters (replicates)
    trials = sorted(df_setting['trial_idx'].unique())
    n_trials = len(trials)

    if n_trials == 0:
        return np.nan

    # Create rater matrix (n_subjects x n_raters)
    rater_matrix = []
    for trial in trials:
        trial_data = df_setting[df_setting['trial_idx'] == trial].sort_values('replicate')
        ratings = [category_map.get(r, 2) for r in trial_data['parsed_output']]

        # Handle missing replicates
        while len(ratings) < 3:
            ratings.append(2)  # Treat missing as INVALID

        rater_matrix.append(ratings)

    rater_matrix = np.array(rater_matrix)

    try:
        # Use statsmodels aggregate_raters to convert to format needed by fleiss_kappa
        # aggregate_raters expects (n_subjects, n_raters) and returns (n_subjects, n_categories)
        aggregated, _ = aggregate_raters(rater_matrix)
        kappa = fleiss_kappa(aggregated, method='fleiss')
        return kappa
    except Exception as e:
        print(f"Warning: Could not compute Fleiss' kappa: {e}")
        return np.nan


def analyze_predictions(predictions_path: Path, output_dir: Path):
    """Main analysis function."""

    # Load predictions
    df = pd.read_csv(predictions_path)
    print(f"Loaded {len(df)} predictions")

    # Get unique settings
    settings_cols = ['vendor', 'model', 'temperature', 'reasoning_level']
    settings = df[settings_cols].drop_duplicates()
    print(f"Found {len(settings)} unique settings")

    # Results storage
    metrics_rows = []
    majority_vote_rows = []

    for _, setting in settings.iterrows():
        vendor = setting['vendor']
        model = setting['model']
        temp = setting['temperature']
        level = setting['reasoning_level']

        # Filter to this setting
        mask = (
            (df['vendor'] == vendor) &
            (df['model'] == model) &
            (df['reasoning_level'] == level)
        )

        # Handle temperature comparison (NaN safe)
        if pd.isna(temp):
            mask = mask & (df['temperature'].isna())
        else:
            mask = mask & (df['temperature'] == temp)

        df_setting = df[mask].copy()

        if len(df_setting) == 0:
            continue

        # Compute Fleiss' kappa
        kappa = compute_fleiss_kappa_for_setting(df_setting)

        # Compute majority votes per trial
        trials = sorted(df_setting['trial_idx'].unique())
        majority_votes = []
        ground_truths = []

        for trial in trials:
            trial_data = df_setting[df_setting['trial_idx'] == trial]
            predictions = trial_data['parsed_output'].tolist()
            majority = compute_majority_vote(predictions)
            gt = trial_data['ground_truth'].iloc[0]

            majority_votes.append(majority)
            ground_truths.append(gt)

            majority_vote_rows.append({
                'trial_idx': trial,
                'vendor': vendor,
                'model': model,
                'temperature': temp,
                'reasoning_level': level,
                'majority_vote': majority,
                'ground_truth': gt
            })

        # Compute per-replicate F1 scores and track valid prediction rates
        f1_per_replicate = []
        n_valid_per_replicate = []
        pct_valid_per_replicate = []
        n_total_trials = len(trials)

        for rep in [1, 2, 3]:
            rep_data = df_setting[df_setting['replicate'] == rep]
            rep_preds = []
            rep_gts = []

            for trial in trials:
                trial_rep = rep_data[rep_data['trial_idx'] == trial]
                if len(trial_rep) > 0:
                    pred = trial_rep['parsed_output'].iloc[0]
                    gt = trial_rep['ground_truth'].iloc[0]

                    # Only include valid predictions
                    if pred in ("POSITIVE", "NEGATIVE") and gt in ("POSITIVE", "NEGATIVE"):
                        rep_preds.append(pred)
                        rep_gts.append(gt)

            n_valid = len(rep_preds)
            pct_valid = (n_valid / n_total_trials * 100) if n_total_trials > 0 else 0.0
            n_valid_per_replicate.append(n_valid)
            pct_valid_per_replicate.append(pct_valid)

            if len(rep_preds) > 0:
                f1_rep = f1_score(rep_gts, rep_preds, average='macro', labels=["POSITIVE", "NEGATIVE"])
                f1_per_replicate.append(f1_rep)
            else:
                f1_per_replicate.append(np.nan)

        # Compute statistics across replicates for F1
        valid_f1s = [f for f in f1_per_replicate if not np.isnan(f)]
        if len(valid_f1s) > 0:
            f1_mean = np.mean(valid_f1s)
            f1_std = np.std(valid_f1s, ddof=1) if len(valid_f1s) > 1 else 0.0
            f1_min = np.min(valid_f1s)
            f1_max = np.max(valid_f1s)
        else:
            f1_mean = f1_std = f1_min = f1_max = np.nan

        # Compute statistics for valid prediction percentages
        pct_valid_mean = np.mean(pct_valid_per_replicate)
        pct_valid_min = np.min(pct_valid_per_replicate)
        pct_valid_max = np.max(pct_valid_per_replicate)

        # Compute metrics
        n_invalid = sum(1 for v in majority_votes if v in ("INVALID", "ERROR"))
        n_error = sum(1 for _, row in df_setting.iterrows() if row['parsed_output'] == "ERROR")

        # Filter for valid predictions for F1/accuracy (majority vote)
        valid_mask = [v in ("POSITIVE", "NEGATIVE") and g in ("POSITIVE", "NEGATIVE")
                      for v, g in zip(majority_votes, ground_truths)]
        valid_preds = [v for v, m in zip(majority_votes, valid_mask) if m]
        valid_gts = [g for g, m in zip(ground_truths, valid_mask) if m]

        if len(valid_preds) > 0:
            accuracy = accuracy_score(valid_gts, valid_preds)
            f1_macro = f1_score(valid_gts, valid_preds, average='macro', labels=["POSITIVE", "NEGATIVE"])
        else:
            accuracy = np.nan
            f1_macro = np.nan

        metrics_rows.append({
            'vendor': vendor,
            'model': model,
            'temperature': temp,
            'reasoning_level': level,
            'fleiss_kappa': kappa,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_rep1': f1_per_replicate[0],
            'f1_rep2': f1_per_replicate[1],
            'f1_rep3': f1_per_replicate[2],
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'f1_min': f1_min,
            'f1_max': f1_max,
            'n_valid_rep1': n_valid_per_replicate[0],
            'n_valid_rep2': n_valid_per_replicate[1],
            'n_valid_rep3': n_valid_per_replicate[2],
            'pct_valid_rep1': pct_valid_per_replicate[0],
            'pct_valid_rep2': pct_valid_per_replicate[1],
            'pct_valid_rep3': pct_valid_per_replicate[2],
            'pct_valid_mean': pct_valid_mean,
            'pct_valid_min': pct_valid_min,
            'pct_valid_max': pct_valid_max,
            'n_invalid': n_invalid,
            'n_error': n_error,
            'n_trials': len(trials)
        })

        print(f"{vendor} temp={temp} level={level}: kappa={kappa:.3f}, F1_majority={f1_macro:.3f}, F1_mean={f1_mean:.3f}±{f1_std:.3f}, valid={pct_valid_mean:.1f}%, invalid={n_invalid}")

    # Save results
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")

    majority_df = pd.DataFrame(majority_vote_rows)
    majority_path = output_dir / "majority_votes.csv"
    majority_df.to_csv(majority_path, index=False)
    print(f"Saved majority votes to {majority_path}")

    # Generate vendor-specific tables for paper
    generate_vendor_tables(metrics_df, output_dir)

    # Print summary
    print("\n=== Summary ===")
    print(f"Total settings analyzed: {len(metrics_df)}")
    print(f"\nBest F1 scores (majority vote):")
    top_f1 = metrics_df.nlargest(5, 'f1_macro')[['vendor', 'temperature', 'reasoning_level', 'f1_macro', 'f1_mean', 'f1_std', 'pct_valid_mean', 'fleiss_kappa']]
    print(top_f1.to_string(index=False))

    print(f"\nBest mean F1 scores (across replicates):")
    top_f1_mean = metrics_df.nlargest(5, 'f1_mean')[['vendor', 'temperature', 'reasoning_level', 'f1_mean', 'f1_std', 'f1_min', 'f1_max', 'pct_valid_mean']]
    print(top_f1_mean.to_string(index=False))

    print(f"\nHighest agreement (Fleiss' kappa):")
    top_kappa = metrics_df.nlargest(5, 'fleiss_kappa')[['vendor', 'temperature', 'reasoning_level', 'fleiss_kappa', 'f1_mean', 'f1_std', 'pct_valid_mean']]
    print(top_kappa.to_string(index=False))

    print(f"\nLowest valid prediction rates:")
    lowest_valid = metrics_df.nsmallest(5, 'pct_valid_mean')[['vendor', 'temperature', 'reasoning_level', 'pct_valid_mean', 'pct_valid_min', 'pct_valid_max', 'f1_mean']]
    print(lowest_valid.to_string(index=False))


def generate_vendor_tables(metrics_df: pd.DataFrame, output_dir: Path):
    """
    Generate vendor-specific tables for paper with formatted metrics.

    Args:
        metrics_df: DataFrame with computed metrics
        output_dir: Directory to save tables
    """
    # Define sort orders for reasoning levels (capitalized for display)
    reasoning_order = {
        'openai': ['None', 'Low', 'Medium', 'High', 'Xhigh'],
        'gemini': ['Minimal', 'Low', 'Medium', 'High']
    }

    for vendor in ['openai', 'gemini']:
        # Filter to this vendor
        vendor_df = metrics_df[metrics_df['vendor'] == vendor].copy()

        if len(vendor_df) == 0:
            print(f"Warning: No data found for vendor {vendor}")
            continue

        # Create output DataFrame with required columns and publication-ready headers
        table_rows = []
        for _, row in vendor_df.iterrows():
            # Format F1 range
            f1_range = f"{row['f1_min']:.3f} - {row['f1_max']:.3f}"

            # Calculate invalid percentage
            invalid_pct = 100 - row['pct_valid_mean']

            # Format reasoning level to title case
            reasoning_formatted = str(row['reasoning_level']).capitalize()

            table_row = {
                'Reasoning level': reasoning_formatted,
                'Temperature': row['temperature'],
                "Fleiss' Kappa": round(row['fleiss_kappa'], 3),
                'F1 mean': round(row['f1_mean'], 3),
                'F1 range': f1_range,
                'F1 majority': round(row['f1_macro'], 3),
                'Invalid [%]': round(invalid_pct, 1)
            }
            table_rows.append(table_row)

        table_df = pd.DataFrame(table_rows)

        # Sort by reasoning level (custom order), then temperature
        # Create categorical type for reasoning level with proper order
        table_df['Reasoning level'] = pd.Categorical(
            table_df['Reasoning level'],
            categories=reasoning_order[vendor],
            ordered=True
        )

        # Sort: first by reasoning level, then by temperature (NaN first for OpenAI)
        table_df = table_df.sort_values(['Reasoning level', 'Temperature'], na_position='first')

        # Save to CSV
        output_path = output_dir / f"table_{vendor}.csv"
        table_df.to_csv(output_path, index=False)
        print(f"Saved {vendor} table to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM reproducibility results")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predictions.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    predictions_path = Path(args.predictions) if args.predictions else script_dir / "results" / "predictions.csv"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "results"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    if not predictions_path.exists():
        print(f"Error: Predictions file not found: {predictions_path}")
        print("Run run_evaluation.py first to generate predictions.")
        return 1

    analyze_predictions(predictions_path, output_dir)
    return 0


if __name__ == "__main__":
    exit(main())
