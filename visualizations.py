#!/usr/bin/env python3
"""
Visualization scripts for LLM Reproducibility Analysis

Suggested visualizations:
1. Confusion matrices for largest F1 range settings (3 replicates each)
2. F1 score distributions by vendor/temperature/reasoning level
3. Reproducibility heatmaps (Fleiss' kappa by settings)
4. Temperature vs F1 performance curves
5. Reasoning level comparison charts
6. Invalid prediction rates by setting
7. Scatter plots: F1 mean vs Fleiss' kappa (reproducibility-performance tradeoff)
8. Violin plots: F1 distributions across replicates
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Set Arial as default font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']


def get_vendor_display_name(vendor):
    """Map internal vendor names to display names."""
    mapping = {
        'openai': 'GPT-5.2',
        'gemini': 'Gemini 3 Flash Preview'
    }
    return mapping.get(vendor, vendor.upper())


def plot_confusion_matrices_largest_f1_range(predictions_df, metrics_df, output_dir):
    """
    Plot confusion matrices for the settings with largest F1 ranges.
    Creates 3x2 grid: 3 replicates for each vendor (OpenAI and Gemini).
    """
    # Calculate F1 range width
    metrics_df['f1_range_width'] = metrics_df['f1_max'] - metrics_df['f1_min']

    # Find largest F1 range for each vendor
    openai_setting = metrics_df[metrics_df['vendor'] == 'openai'].nlargest(1, 'f1_range_width').iloc[0]
    gemini_setting = metrics_df[metrics_df['vendor'] == 'gemini'].nlargest(1, 'f1_range_width').iloc[0]

    settings = [
        ('openai', openai_setting),
        ('gemini', gemini_setting)
    ]

    # Create figure with 3 rows (replicates) x 2 cols (vendors)
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle('Confusion Matrices: Largest F1 Range Settings', fontsize=16, fontweight='bold', y=0.995)

    for col_idx, (vendor, setting) in enumerate(settings):
        temp = setting['temperature']
        level = setting['reasoning_level']

        # Filter predictions for this setting
        mask = (
            (predictions_df['vendor'] == vendor) &
            (predictions_df['model'] == setting['model']) &
            (predictions_df['reasoning_level'] == level)
        )

        if pd.isna(temp):
            mask = mask & (predictions_df['temperature'].isna())
        else:
            mask = mask & (predictions_df['temperature'] == temp)

        setting_df = predictions_df[mask].copy()

        # Plot confusion matrix for each replicate
        for rep_idx, replicate in enumerate([1, 2, 3]):
            ax = axes[rep_idx, col_idx]

            rep_df = setting_df[setting_df['replicate'] == replicate]

            # Filter valid predictions only
            valid_mask = (
                rep_df['parsed_output'].isin(['POSITIVE', 'NEGATIVE']) &
                rep_df['ground_truth'].isin(['POSITIVE', 'NEGATIVE'])
            )
            valid_df = rep_df[valid_mask]

            if len(valid_df) > 0:
                y_true = valid_df['ground_truth']
                y_pred = valid_df['parsed_output']

                # Compute confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=['POSITIVE', 'NEGATIVE'])

                # Calculate F1 for this replicate
                from sklearn.metrics import f1_score
                f1 = f1_score(y_true, y_pred, average='macro', labels=['POSITIVE', 'NEGATIVE'])

                # Plot
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Positive', 'Negative'])
                disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')

                # Title
                if rep_idx == 0:
                    temp_str = f'T={temp}' if not pd.isna(temp) else 'T=None'
                    level_str = level.capitalize() if isinstance(level, str) else level
                    title = f'{get_vendor_display_name(vendor)}\n{temp_str}, {level_str}\nRep {replicate} (F1={f1:.3f})'
                else:
                    title = f'Replicate {replicate}\n(F1={f1:.3f})'

                ax.set_title(title, fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No valid predictions', ha='center', va='center')
                ax.set_title(f'Replicate {replicate}', fontsize=10)

            # Clean up labels
            if rep_idx < 2:
                ax.set_xlabel('')
            if col_idx > 0:
                ax.set_ylabel('')

    plt.tight_layout()
    output_path = output_dir / 'confusion_matrices_largest_f1_range.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def plot_f1_by_temperature(metrics_df, output_dir):
    """Plot F1 mean vs temperature for both vendors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, vendor in enumerate(['openai', 'gemini']):
        ax = axes[idx]
        vendor_df = metrics_df[metrics_df['vendor'] == vendor].copy()

        # Group by temperature (exclude NaN for OpenAI reasoning modes)
        temp_df = vendor_df[vendor_df['temperature'].notna()].copy()

        if vendor == 'gemini':
            # Gemini: separate line for each thinking level
            for thinking in ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH']:
                subset = temp_df[temp_df['reasoning_level'] == thinking]
                if len(subset) > 0:
                    subset = subset.sort_values('temperature')
                    ax.plot(subset['temperature'], subset['f1_mean'],
                           marker='o', label=thinking.capitalize(), linewidth=2)
                    ax.fill_between(subset['temperature'],
                                   subset['f1_mean'] - subset['f1_std'],
                                   subset['f1_mean'] + subset['f1_std'],
                                   alpha=0.2)
        else:
            # OpenAI: single line for temperature sweep (reasoning=none)
            subset = temp_df[temp_df['reasoning_level'] == 'none'].sort_values('temperature')
            ax.plot(subset['temperature'], subset['f1_mean'],
                   marker='o', linewidth=2, color='C0', label='No reasoning')
            ax.fill_between(subset['temperature'],
                           subset['f1_mean'] - subset['f1_std'],
                           subset['f1_mean'] + subset['f1_std'],
                           alpha=0.2, color='C0')

        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('F1 Score (mean ± std)', fontsize=12)
        ax.set_title(f'{get_vendor_display_name(vendor)}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.90, 0.98)

    plt.tight_layout()
    output_path = output_dir / 'f1_vs_temperature.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def plot_reasoning_levels_comparison(metrics_df, output_dir):
    """Compare F1 scores across reasoning/thinking levels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # OpenAI reasoning levels (temperature-free)
    ax = axes[0]
    openai_df = metrics_df[
        (metrics_df['vendor'] == 'openai') &
        (metrics_df['temperature'].isna())
    ].copy()

    reasoning_order = ['none', 'low', 'medium', 'high', 'xhigh']
    openai_df['reasoning_level'] = pd.Categorical(openai_df['reasoning_level'],
                                                   categories=reasoning_order, ordered=True)
    openai_df = openai_df.sort_values('reasoning_level')

    ax.bar(range(len(openai_df)), openai_df['f1_mean'],
           yerr=openai_df['f1_std'], capsize=5, alpha=0.7)
    ax.set_xticks(range(len(openai_df)))
    ax.set_xticklabels([r.capitalize() for r in openai_df['reasoning_level']], rotation=45)
    ax.set_ylabel('F1 Score (mean ± std)', fontsize=12)
    ax.set_title('GPT-5.2', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.90, 0.98)

    # Gemini thinking levels (average across temperatures)
    ax = axes[1]
    gemini_df = metrics_df[metrics_df['vendor'] == 'gemini'].copy()

    # Average across temperatures for each thinking level
    thinking_stats = gemini_df.groupby('reasoning_level').agg({
        'f1_mean': 'mean',
        'f1_std': 'mean'
    }).reset_index()

    thinking_order = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH']
    thinking_stats['reasoning_level'] = pd.Categorical(thinking_stats['reasoning_level'],
                                                       categories=thinking_order, ordered=True)
    thinking_stats = thinking_stats.sort_values('reasoning_level')

    ax.bar(range(len(thinking_stats)), thinking_stats['f1_mean'],
           yerr=thinking_stats['f1_std'], capsize=5, alpha=0.7, color='C1')
    ax.set_xticks(range(len(thinking_stats)))
    ax.set_xticklabels([t.capitalize() for t in thinking_stats['reasoning_level']], rotation=45)
    ax.set_ylabel('F1 Score (mean ± std, averaged over temps)', fontsize=12)
    ax.set_title(f'{get_vendor_display_name("gemini")}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.90, 0.98)

    plt.tight_layout()
    output_path = output_dir / 'reasoning_levels_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def plot_reproducibility_heatmap(metrics_df, output_dir):
    """Heatmap of Fleiss' kappa by temperature and reasoning level."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, vendor in enumerate(['openai', 'gemini']):
        ax = axes[idx]
        vendor_df = metrics_df[metrics_df['vendor'] == vendor].copy()

        if vendor == 'gemini':
            # Gemini: 2D heatmap (temperature x thinking level)
            pivot = vendor_df.pivot_table(
                values='fleiss_kappa',
                index='reasoning_level',
                columns='temperature'
            )

            # Reorder
            thinking_order = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH']
            pivot = pivot.reindex(thinking_order)

            # Capitalize the index labels
            pivot.index = [level.capitalize() for level in pivot.index]

            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                       vmin=0.9, vmax=1.0, ax=ax, cbar_kws={'label': "Fleiss' Kappa"})
            ax.set_ylabel('Thinking Level', fontsize=12)
            ax.set_xlabel('Temperature', fontsize=12)
        else:
            # OpenAI: split into two sections (temp sweep vs reasoning levels)
            # Create a combined view
            temp_df = vendor_df[vendor_df['temperature'].notna()].copy()
            reason_df = vendor_df[vendor_df['temperature'].isna()].copy()

            # Create data for heatmap (vertical bars)
            data = []
            labels = []

            for _, row in temp_df.sort_values('temperature').iterrows():
                data.append([row['fleiss_kappa']])
                labels.append(f"T={row['temperature']:.1f}")

            for _, row in reason_df.iterrows():
                data.append([row['fleiss_kappa']])
                labels.append(row['reasoning_level'].capitalize())

            sns.heatmap(np.array(data), annot=True, fmt='.3f', cmap='RdYlGn',
                       vmin=0.9, vmax=1.0, ax=ax, yticklabels=labels,
                       xticklabels=['Fleiss Kappa'], cbar_kws={'label': "Fleiss' Kappa"})
            ax.set_ylabel('Setting', fontsize=12)

        ax.set_title(f'{get_vendor_display_name(vendor)}: Reproducibility', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'reproducibility_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def plot_f1_vs_kappa_scatter(metrics_df, output_dir):
    """Scatter plot: F1 performance vs reproducibility (Fleiss' kappa)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for vendor, marker, color in [('openai', 'o', 'C0'), ('gemini', 's', 'C1')]:
        vendor_df = metrics_df[metrics_df['vendor'] == vendor]
        ax.scatter(vendor_df['fleiss_kappa'], vendor_df['f1_mean'],
                  s=100, alpha=0.6, marker=marker, label=get_vendor_display_name(vendor), color=color)

    ax.set_xlabel("Fleiss' Kappa (Reproducibility)", fontsize=12)
    ax.set_ylabel('F1 Mean (Performance)', fontsize=12)
    ax.set_title('Performance vs Reproducibility Trade-off', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'f1_vs_kappa_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def plot_invalid_rates_by_setting(metrics_df, output_dir):
    """Bar chart of invalid prediction percentages by setting."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for idx, vendor in enumerate(['openai', 'gemini']):
        ax = axes[idx]
        vendor_df = metrics_df[metrics_df['vendor'] == vendor].copy()

        # Calculate invalid percentage (100 - pct_valid_mean)
        vendor_df['invalid_pct'] = 100 - vendor_df['pct_valid_mean']

        # Create labels
        vendor_df['setting_label'] = vendor_df.apply(
            lambda row: f"T={row['temperature']:.1f}, {row['reasoning_level'].capitalize()}"
            if not pd.isna(row['temperature'])
            else row['reasoning_level'].capitalize(),
            axis=1
        )

        vendor_df = vendor_df.sort_values('invalid_pct', ascending=False)

        ax.barh(range(len(vendor_df)), vendor_df['invalid_pct'], alpha=0.7)
        ax.set_yticks(range(len(vendor_df)))
        ax.set_yticklabels(vendor_df['setting_label'], fontsize=9)
        ax.set_xlabel('Invalid Predictions (%)', fontsize=12)
        ax.set_title(f'{get_vendor_display_name(vendor)}: Invalid Prediction Rates', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

    plt.tight_layout()
    output_path = output_dir / 'invalid_rates_by_setting.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def plot_f1_distributions_violin(predictions_df, metrics_df, output_dir):
    """Violin plots showing F1 distribution across replicates."""
    # Calculate F1 for each (setting, replicate) combination
    f1_data = []

    for _, setting in metrics_df.iterrows():
        vendor = setting['vendor']
        temp = setting['temperature']
        level = setting['reasoning_level']

        # Filter predictions
        mask = (
            (predictions_df['vendor'] == vendor) &
            (predictions_df['reasoning_level'] == level)
        )

        if pd.isna(temp):
            mask = mask & (predictions_df['temperature'].isna())
        else:
            mask = mask & (predictions_df['temperature'] == temp)

        setting_df = predictions_df[mask]

        for rep in [1, 2, 3]:
            rep_df = setting_df[setting_df['replicate'] == rep]

            # Valid predictions only
            valid_mask = (
                rep_df['parsed_output'].isin(['POSITIVE', 'NEGATIVE']) &
                rep_df['ground_truth'].isin(['POSITIVE', 'NEGATIVE'])
            )
            valid_df = rep_df[valid_mask]

            if len(valid_df) > 0:
                from sklearn.metrics import f1_score
                f1 = f1_score(valid_df['ground_truth'], valid_df['parsed_output'],
                            average='macro', labels=['POSITIVE', 'NEGATIVE'])

                f1_data.append({
                    'vendor': vendor,
                    'f1': f1,
                    'temperature': temp,
                    'reasoning_level': level
                })

    f1_df = pd.DataFrame(f1_data)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, vendor in enumerate(['openai', 'gemini']):
        ax = axes[idx]
        vendor_f1 = f1_df[f1_df['vendor'] == vendor]

        sns.violinplot(data=vendor_f1, y='f1', ax=ax, color='skyblue')
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title(f'{get_vendor_display_name(vendor)}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0.90, 0.98)

    plt.tight_layout()
    output_path = output_dir / 'f1_distributions_violin.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def create_combined_figure(plots_dir, output_dir):
    """Create a combined figure with three key plots stacked vertically."""
    from PIL import Image

    # Load the three images
    violin_path = plots_dir / 'f1_distributions_violin.png'
    temp_path = plots_dir / 'f1_vs_temperature.png'
    reasoning_path = plots_dir / 'reasoning_levels_comparison.png'

    # Check if files exist
    if not all([violin_path.exists(), temp_path.exists(), reasoning_path.exists()]):
        print('Warning: Cannot create combined figure - some plots are missing')
        return

    # Load images
    violin_img = Image.open(violin_path)
    temp_img = Image.open(temp_path)
    reasoning_img = Image.open(reasoning_path)

    # Create figure with 3 rows
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    # Display images
    axes[0].imshow(violin_img)
    axes[0].axis('off')
    axes[0].set_title('a) F1 Score Distribution Across All Settings',
                      fontsize=16, fontweight='bold', loc='left', pad=10, fontfamily='Arial')

    axes[1].imshow(temp_img)
    axes[1].axis('off')
    axes[1].set_title('b) F1 Score vs Temperature',
                      fontsize=16, fontweight='bold', loc='left', pad=10, fontfamily='Arial')

    axes[2].imshow(reasoning_img)
    axes[2].axis('off')
    axes[2].set_title('c) Reasoning/Thinking Levels Comparison',
                      fontsize=16, fontweight='bold', loc='left', pad=10, fontfamily='Arial')

    plt.tight_layout()
    output_path = output_dir / 'combined_key_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def main():
    """Generate all visualizations."""
    script_dir = Path(__file__).parent

    # Load data
    predictions_df = pd.read_csv(script_dir / 'results' / 'predictions.csv')
    metrics_df = pd.read_csv(script_dir / 'results' / 'metrics.csv')

    # Output directory for plots
    plots_dir = script_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print('Generating visualizations...\n')

    # Create all plots
    plot_confusion_matrices_largest_f1_range(predictions_df, metrics_df, plots_dir)
    plot_f1_by_temperature(metrics_df, plots_dir)
    plot_reasoning_levels_comparison(metrics_df, plots_dir)
    plot_reproducibility_heatmap(metrics_df, plots_dir)
    plot_f1_vs_kappa_scatter(metrics_df, plots_dir)
    plot_invalid_rates_by_setting(metrics_df, plots_dir)
    plot_f1_distributions_violin(predictions_df, metrics_df, plots_dir)

    # Create combined figure
    print('\nCreating combined figure...')
    create_combined_figure(plots_dir, plots_dir)

    print('\n✓ All visualizations generated successfully!')


if __name__ == '__main__':
    main()
