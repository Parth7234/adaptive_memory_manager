"""
Visualization module v2 for benchmark results.
Adds: energy profiling chart, tiered preloading breakdown, memory pressure comparison.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

C = {
    'primary': '#58a6ff', 'success': '#3fb950', 'danger': '#f85149',
    'warning': '#d29922', 'purple': '#bc8cff', 'cyan': '#39d2c0',
    'orange': '#f0883e', 'pink': '#f778ba',
}


def _bar_labels(ax, bars, fmt='{:.1f}', suffix='', offset=0.5):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+offset,
                fmt.format(b.get_height())+suffix,
                ha='center', fontsize=10, fontweight='bold', color='#c9d1d9')


def plot_training_curves(history_path, save_dir):
    with open(history_path) as f:
        h = json.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(h['train_loss']) + 1)

    axes[0].plot(epochs, h['train_loss'], color=C['primary'], linewidth=2, label='Train')
    axes[0].plot(epochs, h['val_loss'], color=C['danger'], linewidth=2, label='Validation')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in h['val_acc']], color=C['success'], linewidth=2, label='Top-1')
    axes[1].plot(epochs, [a*100 for a in h['val_top3_acc']], color=C['purple'], linewidth=2, label='Top-3')
    axes[1].axhline(y=75, color=C['warning'], linestyle='--', alpha=0.7, label='Target (75%)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Prediction Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    if len(h['val_acc']) > 1:
        deltas = [h['val_acc'][i]-h['val_acc'][i-1] for i in range(1, len(h['val_acc']))]
        axes[2].bar(range(2, len(h['val_acc'])+1), [d*100 for d in deltas],
                    color=[C['success'] if d > 0 else C['danger'] for d in deltas], alpha=0.8)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Delta Accuracy (%)')
    axes[2].set_title('Epoch-over-Epoch Improvement'); axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='#8b949e', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_curves.png")


def plot_kpi_comparison(results_path, save_dir):
    with open(results_path) as f:
        r = json.load(f)
    lru, adp = r['lru'], r['adaptive']

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # 1. Cache Hit Rate
    ax = axes[0, 0]
    bars = ax.bar(['LRU', 'Adaptive'], [lru['hit_rate']*100, adp['hit_rate']*100],
                  color=[C['danger'], C['success']], width=0.5, alpha=0.85)
    ax.axhline(y=85, color=C['warning'], linestyle='--', alpha=0.7, label='Target: 85%')
    ax.set_ylabel('Hit Rate (%)'); ax.set_title('Cache Hit Rate'); ax.legend()
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3, axis='y')
    _bar_labels(ax, bars, suffix='%', offset=1)

    # 2. Average Load Time
    ax = axes[0, 1]
    bars = ax.bar(['LRU', 'Adaptive'], [lru['avg_load_time_ms'], adp['avg_load_time_ms']],
                  color=[C['danger'], C['success']], width=0.5, alpha=0.85)
    ax.set_ylabel('Avg Load Time (ms)'); ax.set_title('Application Load Time')
    ax.grid(True, alpha=0.3, axis='y')
    _bar_labels(ax, bars, suffix='ms', offset=0.2)

    # 3. Thrashing Events
    ax = axes[0, 2]
    bars = ax.bar(['LRU', 'Adaptive'], [lru['thrashing_events'], adp['thrashing_events']],
                  color=[C['danger'], C['success']], width=0.5, alpha=0.85)
    ax.set_ylabel('Thrashing Events'); ax.set_title('Memory Thrashing')
    ax.grid(True, alpha=0.3, axis='y')
    _bar_labels(ax, bars, fmt='{:.0f}', offset=0.3)

    # 4. Page Faults
    ax = axes[1, 0]
    bars = ax.bar(['LRU', 'Adaptive'], [lru['page_faults'], adp['page_faults']],
                  color=[C['danger'], C['success']], width=0.5, alpha=0.85)
    ax.set_ylabel('Page Faults'); ax.set_title('Page Faults (Cache Misses)')
    ax.grid(True, alpha=0.3, axis='y')
    _bar_labels(ax, bars, fmt='{:.0f}', offset=0.3)

    # 5. Energy Consumption
    ax = axes[1, 1]
    lru_e = lru.get('energy_total_mj', 0)
    adp_e = adp.get('energy_total_mj', 0)
    bars = ax.bar(['LRU', 'Adaptive'], [lru_e, adp_e],
                  color=[C['danger'], C['success']], width=0.5, alpha=0.85)
    ax.set_ylabel('Energy (mJ)'); ax.set_title('Battery / Energy Consumption')
    ax.grid(True, alpha=0.3, axis='y')
    _bar_labels(ax, bars, fmt='{:.0f}', suffix='mJ', offset=max(lru_e, adp_e)*0.02+1)

    # 6. Tiered Preloading Breakdown
    ax = axes[1, 2]
    tier_labels = ['Tier 1\n(Metadata)', 'Tier 2\n(BG Cache)', 'Tier 3\n(Full RAM)']
    tier_vals = [adp.get('preload_tier1_hits', 0),
                 adp.get('preload_tier2_hits', 0),
                 adp.get('preload_tier3_hits', 0)]
    bars = ax.bar(tier_labels, tier_vals,
                  color=[C['warning'], C['orange'], C['cyan']], width=0.5, alpha=0.85)
    ax.set_ylabel('Preload Hits'); ax.set_title('Tiered Pre-Loading Hits')
    ax.grid(True, alpha=0.3, axis='y')
    _bar_labels(ax, bars, fmt='{:.0f}', offset=0.3)

    plt.suptitle('Context-Aware Adaptive Memory Management v2 — KPI Dashboard',
                 fontsize=16, fontweight='bold', color='#58a6ff', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kpi_dashboard.png'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: kpi_dashboard.png")


def plot_prediction_analysis(results_path, save_dir):
    with open(results_path) as f:
        r = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prediction accuracy
    ax = axes[0]
    acc = r['prediction_accuracy'] * 100
    t3 = r.get('prediction_top3_accuracy', 0) * 100
    random_baseline = (1/20) * 100

    bars = ax.bar(['Random\nBaseline', 'Ensemble\nTop-1', 'Ensemble\nTop-3', 'Target'],
                  [random_baseline, acc, t3, 75],
                  color=[C['danger'], C['primary'], C['success'], C['warning']],
                  width=0.5, alpha=0.85)
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Next-App Prediction Accuracy')
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3, axis='y')
    _bar_labels(ax, bars, suffix='%', offset=1)

    # Energy breakdown
    ax = axes[1]
    lru = r['lru']
    adp = r['adaptive']
    categories = ['Storage\nI/O', 'RAM\nHold', 'Preload', 'Inference', 'Eviction']
    lru_vals = [lru.get('energy_storage_mj', 0), lru.get('energy_ram_hold_mj', 0),
                0, 0, lru.get('energy_eviction_mj', 0)]
    adp_vals = [adp.get('energy_storage_mj', 0), adp.get('energy_ram_hold_mj', 0),
                adp.get('energy_preload_mj', 0), adp.get('energy_inference_mj', 0),
                adp.get('energy_eviction_mj', 0) if 'energy_eviction_mj' in adp else 0]

    x = np.arange(len(categories))
    w = 0.35
    b1 = ax.bar(x - w/2, lru_vals, w, label='LRU', color=C['danger'], alpha=0.8)
    b2 = ax.bar(x + w/2, adp_vals, w, label='Adaptive', color=C['success'], alpha=0.8)
    ax.set_ylabel('Energy (mJ)'); ax.set_title('Energy Breakdown by Category')
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_accuracy.png'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: prediction_accuracy.png")


def plot_pressure_comparison(results_path, pressure_path, save_dir):
    """Compare performance under normal vs memory pressure conditions."""
    if not os.path.exists(pressure_path):
        return
    with open(results_path) as f:
        r4 = json.load(f)
    with open(pressure_path) as f:
        r2 = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Hit rate under pressure
    ax = axes[0]
    labels = ['LRU\n4GB', 'Adaptive\n4GB', 'LRU\n2GB', 'Adaptive\n2GB']
    vals = [r4['lru']['hit_rate']*100, r4['adaptive']['hit_rate']*100,
            r2['lru']['hit_rate']*100, r2['adaptive']['hit_rate']*100]
    colors = [C['danger'], C['success'], C['orange'], C['cyan']]
    bars = ax.bar(labels, vals, color=colors, width=0.5, alpha=0.85)
    ax.axhline(y=85, color=C['warning'], linestyle='--', alpha=0.7, label='Target: 85%')
    ax.set_ylabel('Hit Rate (%)'); ax.set_title('Hit Rate: Normal vs Pressure')
    ax.set_ylim(0, 100); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    _bar_labels(ax, bars, suffix='%', offset=1)

    # Load time under pressure
    ax = axes[1]
    vals = [r4['lru']['avg_load_time_ms'], r4['adaptive']['avg_load_time_ms'],
            r2['lru']['avg_load_time_ms'], r2['adaptive']['avg_load_time_ms']]
    bars = ax.bar(labels, vals, color=colors, width=0.5, alpha=0.85)
    ax.set_ylabel('Load Time (ms)'); ax.set_title('Load Time: Normal vs Pressure')
    ax.grid(True, alpha=0.3, axis='y')
    _bar_labels(ax, bars, suffix='ms', offset=0.3)

    # Thrashing under pressure
    ax = axes[2]
    vals = [r4['lru']['thrashing_events'], r4['adaptive']['thrashing_events'],
            r2['lru']['thrashing_events'], r2['adaptive']['thrashing_events']]
    bars = ax.bar(labels, vals, color=colors, width=0.5, alpha=0.85)
    ax.set_ylabel('Events'); ax.set_title('Thrashing: Normal vs Pressure')
    ax.grid(True, alpha=0.3, axis='y')
    _bar_labels(ax, bars, fmt='{:.0f}', offset=0.3)

    plt.suptitle('Memory Pressure Resilience (4GB vs 2GB RAM)',
                 fontsize=15, fontweight='bold', color='#58a6ff', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pressure_comparison.png'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: pressure_comparison.png")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(base, 'results')
    model_dir = os.path.join(base, 'models')

    print("Generating visualizations...")
    hist_path = os.path.join(model_dir, 'training_history.json')
    res_path = os.path.join(results_dir, 'benchmark_results.json')
    pres_path = os.path.join(results_dir, 'pressure_results.json')

    if os.path.exists(hist_path):
        plot_training_curves(hist_path, results_dir)
    if os.path.exists(res_path):
        plot_kpi_comparison(res_path, results_dir)
        plot_prediction_analysis(res_path, results_dir)
    if os.path.exists(res_path) and os.path.exists(pres_path):
        plot_pressure_comparison(res_path, pres_path, results_dir)

    print("Done!")
