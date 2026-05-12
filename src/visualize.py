"""
Visualization module for benchmark results and training metrics.
Generates publication-ready charts for the KPI dashboard.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json, os

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'text.color': '#c9d1d9', 'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9', 'xtick.color': '#8b949e',
    'ytick.color': '#8b949e', 'grid.color': '#21262d',
    'figure.dpi': 150,
})

COLORS = {
    'primary': '#58a6ff', 'success': '#3fb950', 'danger': '#f85149',
    'warning': '#d29922', 'purple': '#bc8cff', 'cyan': '#39d2c0',
    'orange': '#f0883e', 'pink': '#f778ba',
}


def plot_training_curves(history_path, save_dir):
    """Plot training loss, validation loss, and accuracy curves."""
    with open(history_path) as f:
        h = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(h['train_loss']) + 1)

    # Loss curves
    axes[0].plot(epochs, h['train_loss'], color=COLORS['primary'], linewidth=2, label='Train')
    axes[0].plot(epochs, h['val_loss'], color=COLORS['danger'], linewidth=2, label='Validation')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [a*100 for a in h['val_acc']], color=COLORS['success'], linewidth=2, label='Top-1')
    axes[1].plot(epochs, [a*100 for a in h['val_top3_acc']], color=COLORS['purple'], linewidth=2, label='Top-3')
    axes[1].axhline(y=75, color=COLORS['warning'], linestyle='--', alpha=0.7, label='Target (75%)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Prediction Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Accuracy improvement rate
    if len(h['val_acc']) > 1:
        deltas = [h['val_acc'][i]-h['val_acc'][i-1] for i in range(1, len(h['val_acc']))]
        axes[2].bar(range(2, len(h['val_acc'])+1), [d*100 for d in deltas],
                    color=[COLORS['success'] if d > 0 else COLORS['danger'] for d in deltas], alpha=0.8)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Δ Accuracy (%)')
    axes[2].set_title('Epoch-over-Epoch Improvement'); axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='#8b949e', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_curves.png")


def plot_kpi_comparison(results_path, save_dir):
    """Plot KPI comparison bar chart between LRU and Adaptive."""
    with open(results_path) as f:
        r = json.load(f)

    lru, adp = r['lru'], r['adaptive']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Cache Hit Rate
    ax = axes[0, 0]
    bars = ax.bar(['LRU Baseline', 'Adaptive'], [lru['hit_rate']*100, adp['hit_rate']*100],
                  color=[COLORS['danger'], COLORS['success']], width=0.5, alpha=0.85)
    ax.axhline(y=85, color=COLORS['warning'], linestyle='--', alpha=0.7, label='Target: 85%')
    ax.set_ylabel('Hit Rate (%)'); ax.set_title('Cache Hit Rate'); ax.legend()
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3, axis='y')
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{b.get_height():.1f}%',
                ha='center', fontsize=11, fontweight='bold', color='#c9d1d9')

    # 2. Average Load Time
    ax = axes[0, 1]
    bars = ax.bar(['LRU Baseline', 'Adaptive'], [lru['avg_load_time_ms'], adp['avg_load_time_ms']],
                  color=[COLORS['danger'], COLORS['success']], width=0.5, alpha=0.85)
    ax.set_ylabel('Avg Load Time (ms)'); ax.set_title('Application Load Time')
    ax.grid(True, alpha=0.3, axis='y')
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f'{b.get_height():.1f}ms',
                ha='center', fontsize=11, fontweight='bold', color='#c9d1d9')

    # 3. Thrashing Events
    ax = axes[1, 0]
    bars = ax.bar(['LRU Baseline', 'Adaptive'], [lru['thrashing_events'], adp['thrashing_events']],
                  color=[COLORS['danger'], COLORS['success']], width=0.5, alpha=0.85)
    ax.set_ylabel('Thrashing Events'); ax.set_title('Memory Thrashing (Target: 50%+ reduction)')
    ax.grid(True, alpha=0.3, axis='y')
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{int(b.get_height())}',
                ha='center', fontsize=11, fontweight='bold', color='#c9d1d9')

    # 4. Page Faults
    ax = axes[1, 1]
    bars = ax.bar(['LRU Baseline', 'Adaptive'], [lru['page_faults'], adp['page_faults']],
                  color=[COLORS['danger'], COLORS['success']], width=0.5, alpha=0.85)
    ax.set_ylabel('Page Faults'); ax.set_title('Page Faults (Cache Misses)')
    ax.grid(True, alpha=0.3, axis='y')
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{int(b.get_height())}',
                ha='center', fontsize=11, fontweight='bold', color='#c9d1d9')

    plt.suptitle('Context-Aware Adaptive Memory Management — KPI Dashboard',
                 fontsize=16, fontweight='bold', color='#58a6ff', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kpi_dashboard.png'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: kpi_dashboard.png")


def plot_prediction_analysis(results_path, save_dir):
    """Plot prediction accuracy analysis."""
    with open(results_path) as f:
        r = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    acc = r['prediction_accuracy'] * 100
    random_baseline = (1/20) * 100  # random guess among 20 apps

    bars = ax.bar(['Random Baseline', 'LSTM Predictor', 'Target'],
                  [random_baseline, acc, 75],
                  color=[COLORS['danger'], COLORS['success'], COLORS['warning']],
                  width=0.5, alpha=0.85)

    ax.set_ylabel('Accuracy (%)'); ax.set_title('Next-App Prediction Accuracy')
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3, axis='y')

    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f'{b.get_height():.1f}%',
                ha='center', fontsize=12, fontweight='bold', color='#c9d1d9')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_accuracy.png'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: prediction_accuracy.png")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    results_dir = os.path.join(base, 'results')
    model_dir = os.path.join(base, 'models')

    print("Generating visualizations...")

    history_path = os.path.join(model_dir, 'training_history.json')
    results_path = os.path.join(results_dir, 'benchmark_results.json')

    if os.path.exists(history_path):
        plot_training_curves(history_path, results_dir)
    if os.path.exists(results_path):
        plot_kpi_comparison(results_path, results_dir)
        plot_prediction_analysis(results_path, results_dir)

    print("Done!")
