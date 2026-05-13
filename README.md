# Context-Aware Adaptive Memory Management System v3

**Samsung Problem Statement Solution** — An intelligent on-device memory management system that uses ML-based prediction to optimize application caching, pre-loading, and eviction for smartphones and edge devices.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    User Interaction Layer                     │
│     (App switches, time deltas, context, user profile)       │
└─────────────────────┬────────────────────────────────────────┘
                      │
          ┌───────────▼───────────────┐
          │    Prediction Engine      │
          │    ("The Brain")          │
          │                           │
          │  ┌─────────┐ ┌─────────┐  │
          │  │ T-GRU   │ │ Markov  │  │
          │  │ (ONNX/  │ │ Chain   │  │
          │  │  NPU)   │ │ (CPU)   │  │
          │  └────┬────┘ └────┬────┘  │
          │       └─────┬─────┘       │
          │     Ensemble (30/70)      │
          └───────────┬───────────────┘
                      │ P(next_app)
          ┌───────────▼───────────────┐
          │   Memory Manager v3       │
          │   ("The Brawn")           │
          │                           │
          │  • ARC self-tuning        │
          │    eviction + ghost caches│
          │  • Tiered pre-loading     │
          │    + page clustering      │
          │  • ZRAM compressed tier   │
          │  • Anti-thrashing guard   │
          │  • Energy/battery aware   │
          │  • Token-level KV cache   │
          └───────────────────────────┘
```

## Project Structure

```
samsung/
├── run.py                      # Pipeline orchestrator (7 steps)
├── simulate.py                 # Simulation, benchmarking, ONNX export
├── src/
│   ├── data_generator.py       # Synthetic dataset generator
│   ├── predictor.py            # LSTM + Markov ensemble predictor
│   ├── memory_manager.py       # Adaptive & LRU managers + KV cache v2
│   └── visualize.py            # Dashboard & chart generation
├── data/                       # Generated datasets
│   ├── app_usage_logs.csv      # ~117K app usage records
│   ├── kv_cache_workload.csv   # GenAI KV cache workload
│   └── app_metadata.json       # App definitions
├── models/                     # Trained model weights
│   ├── best_predictor.pth      # Best LSTM checkpoint
│   ├── predictor_edge.onnx     # ONNX model for edge/NPU deployment
│   ├── markov_model.pkl        # Per-user Markov chains
│   └── training_history.json   # Training metrics
└── results/                    # Benchmark results & charts
    ├── benchmark_results.json  # Standard 4GB results
    ├── pressure_results.json   # Memory pressure 2GB results
    ├── kv_cache_results.json   # KV cache stats
    ├── training_curves.png
    ├── kpi_dashboard.png       # 6-panel KPI dashboard
    ├── prediction_accuracy.png # Accuracy + energy breakdown
    └── pressure_comparison.png # 4GB vs 2GB stress test
```

## Quick Start

### Option A: Docker (Recommended — Zero Setup)

```bash
# Build and run the full pipeline in one command
docker build -t edge-memory-sim .
docker run edge-memory-sim

# Or use Docker Compose (exports results to host)
docker compose up --build
```

### Option B: Local Python

```bash
# Install dependencies
pip install torch numpy pandas matplotlib tabulate onnx onnxscript

# Run the full pipeline (7 steps: data → train → simulate → pressure → KV → ONNX → viz)
python3 run.py
```

The pipeline takes ~8 minutes (CPU) or ~6 minutes (MPS/CUDA) and produces all results automatically.

> **Note**: Docker is for **simulation & benchmarking** only. On a real Samsung device, the system runs as a native Android Background Service with NPU inference via Samsung ONE / NNAPI — not in a container. Docker would prevent intercepting real kernel memory paging (`madvise`, `kswapd`, `lmkd`).

## What's New in v3

| Improvement | Description |
|------------|-------------|
| **T-GRU Predictor** | Time-Aware GRU replaces LSTM: time-decay gating, dot-product attention, 25% fewer params (271K → 82KB ONNX) |
| **ARC Eviction** | Adaptive Replacement Cache with ghost caches (B1/B2) self-tunes α/β/γ weights in real-time |
| **ZRAM Compressed Tier** | Evicted apps → LZ4 compressed RAM (0.5ms restore vs 50ms from UFS, 80% hit rate) |
| **Predictive Page Clustering** | Only preload startup page clusters (~25% of app binary), not the full APK |
| **Tiered Pre-loading** | 3-tier confidence system: metadata-warm (>10%), background-cache (>25%), full-RAM (>50%) |
| **Energy/Battery Profiling** | Per-operation mJ tracking with ARM Cortex-A78 power model |
| **Deep KV Cache** | Token-level prefix caching, FP16→INT8 quantization, prefix deduplication |
| **Docker** | One-command reproducibility: `docker build -t edge-memory-sim . && docker run edge-memory-sim` |

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
- **Time-Aware GRU (T-GRU)** with dot-product attention (PyTorch → ONNX):
  - 64-dim app embeddings, 24-dim user embeddings, 128-dim GRU hidden state
  - Time-decay gate: flushes stale context after long idle periods
  - Dot-product attention: looks back at all 10 hidden states
  - Cyclical time encoding (sin/cos of hour and day-of-week)
  - ~271K parameters → **82KB** ONNX model for NPU deployment
- **Per-User Markov Chain**: 2nd-order transition + time-of-day patterns
- **Ensemble**: Weighted combination (30% T-GRU + 70% Markov)

### 3. Memory Manager v3 (`src/memory_manager.py`)

#### ARC Self-Tuning Eviction
- `score = α·recency + β·frequency + γ·ML_prediction` with **self-tuning weights**
- Ghost caches B1/B2 track recently evicted page IDs — hits automatically adapt α/β
- GenAI apps get 1.5× retention bonus; recently-reloaded pages get 2× anti-thrash boost

#### Tiered Pre-loading with Page Clustering
| Confidence Level | Action | Load Time Reduction |
|-----------------|--------|-------------------|
| >10% (Tier 1) | Pre-warm flash storage controller | ~40% faster seek |
| >25% (Tier 2) | Page **startup cluster** into OS cache | ~80% faster load |
| >50% (Tier 3) | Fully map startup cluster into RAM | ~95% instant |

Only the **startup page cluster** (~25% of app binary, the specific 4KB pages needed for first-frame render) is preloaded — not the full APK.

#### ZRAM Compressed RAM Tier
- Evicted apps → LZ4 compressed into 15% RAM reservation (3× space savings)
- Restore from ZRAM: **0.5ms** vs 50ms from UFS storage (100× faster)
- 80% ZRAM hit rate — most evicted apps never touch slow storage

#### Energy-Aware Profiling
Every operation is tracked with millijoule energy costs based on ARM Cortex-A78 / Samsung Exynos power profiles:
- RAM hold: 0.002 mJ/MB/sec (LPDDR5)
- Storage I/O: 0.8 mJ/MB (UFS 4.0 cold read)
- ZRAM compress/decompress: 0.05/0.03 mJ/MB (LZ4)
- NPU inference: 0.15 mJ/call (INT8 T-GRU)

#### Token-Level KV Cache
- **Prefix deduplication**: Shared system prompts across requests reuse same cache
- **3-stage eviction**: Quantize (FP16→INT8) → Offload to storage → Full evict
- **Pinned prefixes**: System prompts always stay in high-speed RAM

### 4. Simulation Engine (`simulate.py`)
- Replays usage traces through LRU and Adaptive managers in parallel
- Standard test (4GB) + Memory pressure test (2GB)
- ONNX model export for edge deployment validation
- Energy profiling per operation category

## Key Design Decisions

1. **Ensemble over single model**: The Markov chain captures per-user deterministic patterns (Calendar→Maps), while the LSTM learns generalizable cross-user temporal features. Together they achieve 80%+ top-3 accuracy.

2. **Anti-thrashing guard**: Pages recently evicted and reloaded get a 2x retention boost, preventing the rapid evict-reload cycle that plagues standard caching.

3. **Tiered pre-loading**: Instead of binary preload, 3 confidence tiers minimize energy cost while maximizing load time improvement. Tier 1 costs only 0.05 mJ/MB vs 0.8 mJ/MB for a cold load.

4. **ARC self-tuning eviction**: `score = α·recency + β·frequency + γ·prediction`, where α/β **automatically adapt** via ghost cache feedback — no manual tuning needed.

5. **ZRAM before storage**: Evicted pages go to compressed RAM first, avoiding 50ms UFS reads in 80% of cases. Net effect: **-41% load time** under 2GB pressure.

6. **Energy budget**: Every operation tracks mJ cost, proving the ML overhead (0.15 mJ/inference) is dwarfed by the storage I/O savings from reduced page faults.

## Sim-to-Real Integration Path

This system is designed as a **drop-in module** for real Android/Linux deployment:

### Android Integration
| Layer | Integration Point | Our Component |
|-------|-------------------|---------------|
| **Kernel** | `madvise()` / `MADV_WILLNEED` | Tier 1/2 pre-loading |
| **Framework** | `lmkd` (Low Memory Killer Daemon) | Eviction scoring |
| **HAL** | NPU HAL (Samsung Exynos NPU) | ONNX LSTM inference |
| **App** | `ActivityManager.getRunningTasks()` | Usage log collection |

### Edge Deployment
```
PyTorch T-GRU (271K params, ~1.1MB)
    → torch.onnx.export() [opset 18]
    → ONNX model (82KB)
    → onnxruntime / Samsung ONE (Neural Engine)
    → INT8 quantization on NPU
    → ~1.5ms inference latency
```

### Linux cgroups Integration
```c
// Pseudo-code for memory.pressure integration
struct adaptive_mm {
    struct ml_predictor *pred;     // ONNX runtime
    struct markov_chain *markov;   // Per-user transition tables
    struct energy_tracker *energy; // Battery-aware decisions
};

// Hook into mm/vmscan.c shrink_page_list()
int adaptive_eviction_score(struct page *page) {
    float ml_prob = predict_next_app(page->app_id);
    float recency = compute_recency(page);
    return ALPHA * recency + BETA * frequency + GAMMA * ml_prob;
}
```

## Requirements
- Python 3.10+
- PyTorch (CPU or MPS/CUDA)
- NumPy, Pandas, matplotlib, tabulate
- onnx, onnxscript for ONNX export
- (Optional) Docker for one-command reproducibility
