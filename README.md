# Context-Aware Adaptive Memory Management System

**Samsung Problem Statement Solution** — An intelligent on-device memory management system that uses ML-based prediction to optimize application caching, pre-loading, and eviction for smartphones and edge devices.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    User Interaction Layer                 │
│          (App switches, timestamps, context)              │
└─────────────────────┬────────────────────────────────────┘
                      │
          ┌───────────▼───────────────┐
          │    Prediction Engine      │
          │    ("The Brain")          │
          │                           │
          │  ┌─────────┐ ┌─────────┐  │
          │  │  LSTM   │ │ Markov  │  │
          │  │ (PyTorch)│ │ Chain   │  │
          │  └────┬────┘ └────┬────┘  │
          │       └─────┬─────┘       │
          │         Ensemble          │
          │     (0.3 LSTM + 0.7 MC)   │
          └───────────┬───────────────┘
                      │ P(next_app)
          ┌───────────▼───────────────┐
          │   Memory Manager          │
          │   ("The Brawn")           │
          │                           │
          │  • Prediction-weighted    │
          │    eviction scoring       │
          │  • Proactive pre-loading  │
          │  • Anti-thrashing guard   │
          │  • KV Cache compression   │
          └───────────────────────────┘
```

## Project Structure

```
samsung/
├── run.py                      # Main orchestration script
├── simulate.py                 # Simulation & benchmarking engine
├── src/
│   ├── data_generator.py       # Synthetic dataset generator
│   ├── predictor.py            # LSTM + Markov ensemble predictor
│   ├── memory_manager.py       # Adaptive & LRU memory managers
│   └── visualize.py            # Result visualization
├── data/                       # Generated datasets
│   ├── app_usage_logs.csv      # ~117K app usage records
│   ├── kv_cache_workload.csv   # GenAI KV cache workload
│   └── app_metadata.json       # App definitions
├── models/                     # Trained model weights
│   ├── best_predictor.pth      # Best LSTM checkpoint
│   ├── markov_model.pkl        # Per-user Markov chains
│   └── training_history.json   # Training metrics
└── results/                    # Benchmark results & charts
    ├── benchmark_results.json
    ├── kv_cache_results.json
    ├── training_curves.png
    ├── kpi_dashboard.png
    └── prediction_accuracy.png
```

## Quick Start

```bash
# Install dependencies
pip install torch numpy pandas scikit-learn matplotlib seaborn tabulate

# Run the full pipeline (data → train → simulate → benchmark → visualize)
python3 run.py
```

The pipeline takes ~5 minutes and produces all results automatically.

## Components

### 1. Data Generator (`src/data_generator.py`)
Generates realistic Android app usage sequences for 50 users over 30 days:
- **20 apps** including 2 GenAI apps (AI Assistant, Image Generator)
- **Temporal patterns**: time-of-day and day-of-week biases
- **Sequential transitions**: Calendar→Maps, Camera→Gallery, etc.
- **User archetypes**: power_user, social_butterfly, casual, content_consumer, professional
- **KV Cache workload**: 500 GenAI inference requests with varying context lengths

### 2. Prediction Engine (`src/predictor.py`)
Hybrid ensemble combining:
- **LSTM with Attention** (PyTorch): User-embedded sequence model with attention mechanism
  - 48-dim app embeddings, 16-dim user embeddings, 128-dim LSTM hidden state
  - Cyclical time encoding (sin/cos of hour and day-of-week)
  - ~299K parameters, lightweight enough for edge deployment
- **Per-User Markov Chain**: 2nd-order transition probabilities + time-of-day patterns
  - Zero training time, instant inference
  - Captures user-specific habits perfectly
- **Ensemble**: Weighted combination (30% LSTM + 70% Markov)

### 3. Memory Manager (`src/memory_manager.py`)
Three components:
- **LRU Baseline**: Standard Least-Recently-Used eviction (comparison baseline)
- **Adaptive Manager**: ML-driven memory management with:
  - Prediction-weighted eviction scoring (recency + frequency + ML prediction)
  - Proactive pre-loading of top-K predicted apps
  - Anti-thrashing protection (prevents rapid evict-reload cycles)
  - GenAI cache awareness (higher priority for expensive models)
- **KV Cache Manager**: Intelligent compression and offloading for GenAI workloads

### 4. Simulation Engine (`simulate.py`)
- Replays real usage traces through both LRU and Adaptive managers
- Uses trained ensemble predictor for real-time probability estimates
- Tracks all KPIs: hit rate, load time, page faults, thrashing, preload hits

## Benchmark Results

| KPI | Target | LRU Baseline | Adaptive | Improvement | Status |
|-----|--------|-------------|----------|-------------|--------|
| Prediction Accuracy (Top-3) | ≥75% | — | **80.2%** | — | ✅ PASS |
| Cache Hit Rate | ≥85% | 98.3% | **98.9%** | +0.6% | ✅ PASS |
| Thrashing Events | 50%+ reduction | 8 | **0** | -100% | ✅ PASS |
| System Stability | 0 issues | — | **0** | — | ✅ PASS |
| Page Faults | Reduction | 443 | **293** | -33.9% | ✅ PASS |
| Evictions | Reduction | 430 | **309** | -28.1% | ✅ PASS |
| Load Time | 20%+ reduction | 8.9ms | **8.1ms** | -9.9% | ⚠️ Close |

**KV Cache Management**: 189 cache reuses, 219 compressions, 18.5GB memory saved.

## Key Design Decisions

1. **Ensemble over single model**: The Markov chain captures per-user deterministic patterns (Calendar→Maps), while the LSTM learns generalizable cross-user temporal features. Together they achieve 80%+ top-3 accuracy.

2. **Anti-thrashing guard**: Pages recently evicted and reloaded get a 2x retention boost, preventing the rapid evict-reload cycle that plagues standard caching.

3. **Conservative pre-loading**: Only pre-loads when probability > 12% AND memory usage < 80%, avoiding the trap of aggressive preloading causing more evictions.

4. **Reward-aligned eviction scoring**: `score = α·recency + β·frequency + γ·prediction`, directly optimizing for the KPI targets (cache hit rate and thrashing reduction).

## Requirements
- Python 3.10+
- PyTorch (CPU or MPS/CUDA)
- NumPy, Pandas, scikit-learn, matplotlib, seaborn, tabulate
