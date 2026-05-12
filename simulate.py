"""
Simulation & Benchmarking Engine v2
Compares Adaptive (ML-driven) vs LRU baseline memory management.
v2: Energy profiling, tiered pre-loading, memory pressure tests, ONNX export.
"""

import numpy as np
import pandas as pd
import torch
import os, json, pickle
from tabulate import tabulate

from src.data_generator import APPS, NUM_APPS
from src.predictor import (NextAppLSTM, MarkovPredictor, EnsemblePredictor, SEQ_LEN)
from src.memory_manager import (LRUMemoryManager, AdaptiveMemoryManager,
                                 KVCacheManager, ENERGY_COSTS)


def load_ensemble(model_dir="models", device=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    model = NextAppLSTM(num_users=50).to(device)
    path = os.path.join(model_dir, "best_predictor.pth")
    if not os.path.exists(path):
        path = os.path.join(model_dir, "final_predictor.pth")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()

    markov = MarkovPredictor(NUM_APPS, order=2)
    mp = os.path.join(model_dir, "markov_model.pkl")
    if os.path.exists(mp):
        with open(mp, "rb") as f:
            data = pickle.load(f)
            markov.user_transitions = data["transitions"]
            markov.user_time_patterns = data["time_patterns"]
            if "app_freq" in data:
                markov.user_app_freq = data["app_freq"]

    return EnsemblePredictor(model, markov, device, lstm_weight=0.3, markov_weight=0.7), device


def run_simulation(df, ensemble, device, total_memory_mb=4096,
                   test_users=None, label=""):
    """Run parallel simulations with LRU baseline and Adaptive manager."""
    if test_users is None:
        users = df["user_id"].unique()
        np.random.seed(99)
        test_users = np.random.choice(users, size=min(10, len(users)), replace=False)

    test_df = df[df["user_id"].isin(test_users)].sort_values(
        ["user_id", "timestamp"]).reset_index(drop=True)

    app_sizes = {int(k): v["memory_mb"] for k, v in APPS.items()}

    lru = LRUMemoryManager(total_memory_mb)
    adaptive = AdaptiveMemoryManager(total_memory_mb, preload_top_k=5)

    pred_correct, pred_top3, pred_total = 0, 0, 0

    suffix = f" [{label}]" if label else ""
    print(f"\n  Simulating {len(test_df):,} events across {len(test_users)} users"
          f" (RAM: {total_memory_mb}MB){suffix}...")

    for uid in test_users:
        user_df = test_df[test_df["user_id"] == uid].reset_index(drop=True)
        if len(user_df) < SEQ_LEN + 1:
            continue

        recent_apps = list(user_df["app_id"].values[:SEQ_LEN])
        timestamps = pd.to_datetime(user_df["timestamp"])
        t0 = timestamps.iloc[0]
        # Compute time deltas for the seed sequence
        recent_deltas = []
        for j in range(SEQ_LEN):
            if j == 0:
                recent_deltas.append(0.0)
            else:
                recent_deltas.append(max(0, (timestamps.iloc[j] - timestamps.iloc[j-1]).total_seconds()))

        for idx in range(SEQ_LEN, len(user_df)):
            row = user_df.iloc[idx]
            app_id = int(row["app_id"])
            size_mb = float(row["memory_mb"])
            hour = int(row["hour"])
            dow = int(row["day_of_week"])
            timestamp = (timestamps.iloc[idx] - t0).total_seconds()
            # Time delta since previous access
            dt = max(0, (timestamps.iloc[idx] - timestamps.iloc[idx-1]).total_seconds())

            probs = ensemble.predict_proba(uid, recent_apps, hour, dow,
                                            time_deltas=recent_deltas)
            predicted = np.argmax(probs)
            top3 = np.argsort(probs)[-3:]

            if predicted == app_id: pred_correct += 1
            if app_id in top3: pred_top3 += 1
            pred_total += 1

            lru.access(app_id, size_mb, timestamp)
            adaptive.access(app_id, size_mb, timestamp,
                          prediction_probs=probs, app_sizes=app_sizes)

            recent_apps.append(app_id)
            recent_deltas.append(dt)
            if len(recent_apps) > SEQ_LEN:
                recent_apps.pop(0)
                recent_deltas.pop(0)

    ls = lru.get_stats()
    as_ = adaptive.get_stats()
    pa = pred_correct / max(1, pred_total)
    pt3 = pred_top3 / max(1, pred_total)

    results = {
        "ram_mb": total_memory_mb,
        "prediction_accuracy": pa,
        "prediction_top3_accuracy": pt3,
        "lru": {
            "avg_load_time_ms": ls.avg_load_time,
            "hit_rate": ls.hit_rate,
            "page_faults": ls.page_faults,
            "thrashing_events": ls.thrashing_events,
            "evictions": ls.evictions,
            "total_accesses": ls.total_accesses,
            "energy_total_mj": ls.energy.total_mj,
            "energy_storage_mj": ls.energy.storage_read_mj,
            "energy_eviction_mj": ls.energy.eviction_mj,
            "energy_ram_hold_mj": ls.energy.ram_hold_mj,
        },
        "adaptive": {
            "avg_load_time_ms": as_.avg_load_time,
            "hit_rate": as_.hit_rate,
            "page_faults": as_.page_faults,
            "thrashing_events": as_.thrashing_events,
            "evictions": as_.evictions,
            "preloads": as_.preloads,
            "preload_hits": as_.preload_hits,
            "preload_tier1_hits": as_.preload_tier1_hits,
            "preload_tier2_hits": as_.preload_tier2_hits,
            "preload_tier3_hits": as_.preload_tier3_hits,
            "zram_hits": as_.zram_hits,
            "zram_stores": as_.zram_stores,
            "ghost_hits_b1": as_.ghost_hits_b1,
            "ghost_hits_b2": as_.ghost_hits_b2,
            "total_accesses": as_.total_accesses,
            "inference_time_ms": as_.total_inference_time_ms,
            "energy_total_mj": as_.energy.total_mj,
            "energy_storage_mj": as_.energy.storage_read_mj,
            "energy_preload_mj": as_.energy.preload_mj,
            "energy_inference_mj": as_.energy.inference_mj,
            "energy_ram_hold_mj": as_.energy.ram_hold_mj,
        },
        "improvements": {}
    }

    imp = results["improvements"]
    if ls.avg_load_time > 0:
        imp["load_time_reduction_pct"] = (
            (ls.avg_load_time - as_.avg_load_time) / ls.avg_load_time * 100)
    if ls.thrashing_events > 0:
        imp["thrashing_reduction_pct"] = (
            (ls.thrashing_events - as_.thrashing_events) / ls.thrashing_events * 100)
    else:
        imp["thrashing_reduction_pct"] = 100.0
    imp["hit_rate_improvement_pct"] = (
        (as_.hit_rate - ls.hit_rate) / max(0.01, ls.hit_rate) * 100)
    if ls.page_faults > 0:
        imp["page_fault_reduction_pct"] = (
            (ls.page_faults - as_.page_faults) / ls.page_faults * 100)
    if ls.energy.total_mj > 0:
        imp["energy_reduction_pct"] = (
            (ls.energy.total_mj - as_.energy.total_mj) / ls.energy.total_mj * 100)

    return results


def run_kv_cache_simulation(kv_df):
    """Simulate KV cache with token-level prefix caching."""
    manager = KVCacheManager(max_kv_memory_mb=2048)
    for _, row in kv_df.iterrows():
        manager.allocate(
            int(row["request_id"]), row["model_type"],
            float(row["kv_cache_size_mb"]), int(row["priority"]),
            bool(row["is_continuation"]),
            context_length=int(row["context_length"]))
    return manager.get_stats()


def export_onnx(model_dir, device):
    """Export the LSTM model to ONNX for edge deployment."""
    try:
        model = NextAppLSTM(num_users=50).to("cpu")
        path = os.path.join(model_dir, "best_predictor.pth")
        if not os.path.exists(path):
            path = os.path.join(model_dir, "final_predictor.pth")
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()

        # Dummy inputs
        dummy_seq = torch.randint(0, 20, (1, SEQ_LEN))
        dummy_ctx = torch.randn(1, 4)
        dummy_uid = torch.zeros(1, 1, dtype=torch.long)

        onnx_path = os.path.join(model_dir, "predictor_edge.onnx")
        torch.onnx.export(
            model, (dummy_seq, dummy_ctx, dummy_uid), onnx_path,
            input_names=["app_sequence", "time_context", "user_id"],
            output_names=["next_app_logits"],
            dynamic_axes={"app_sequence": {0: "batch"}, "time_context": {0: "batch"},
                         "user_id": {0: "batch"}, "next_app_logits": {0: "batch"}},
            opset_version=14,
        )
        # Get file size
        size_kb = os.path.getsize(onnx_path) / 1024
        print(f"  ONNX model exported: {onnx_path} ({size_kb:.0f} KB)")
        return onnx_path, size_kb
    except Exception as e:
        print(f"  ONNX export skipped: {e}")
        return None, 0


def print_results(results, kv_stats, pressure_results=None):
    """Print formatted KPI comparison with energy profiling."""
    print("\n" + "=" * 80)
    print("  CONTEXT-AWARE ADAPTIVE MEMORY MANAGEMENT v2 — BENCHMARK RESULTS")
    print("=" * 80)

    acc = results["prediction_accuracy"]
    t3 = results.get("prediction_top3_accuracy", 0)
    print(f"\n  NEXT-APP PREDICTION MODEL")
    print(f"  Top-1 Accuracy: {acc:.1%}  |  Top-3 Accuracy: {t3:.1%}  (Target: >=75%)")
    if acc >= 0.75:
        print(f"  Status: PASS (Top-1)")
    elif t3 >= 0.75:
        print(f"  Status: PASS (Top-3)")
    else:
        print(f"  Status: {acc:.1%} / {t3:.1%}")

    lru = results["lru"]
    adp = results["adaptive"]
    imp = results["improvements"]

    # ── Memory Management KPIs ──
    print(f"\n  MEMORY MANAGEMENT KPIs (RAM: {results['ram_mb']}MB)")
    rows = [
        ["Cache Hit Rate", f"{lru['hit_rate']:.1%}", f"{adp['hit_rate']:.1%}",
         f"+{imp['hit_rate_improvement_pct']:.1f}%", ">=85%",
         "PASS" if adp['hit_rate'] >= 0.85 else "---"],
        ["Avg Load Time (ms)", f"{lru['avg_load_time_ms']:.1f}", f"{adp['avg_load_time_ms']:.1f}",
         f"-{imp.get('load_time_reduction_pct',0):.1f}%", "20%+ reduction",
         "PASS" if imp.get('load_time_reduction_pct',0) >= 20 else "---"],
        ["Page Faults", str(lru['page_faults']), str(adp['page_faults']),
         f"-{imp.get('page_fault_reduction_pct',0):.1f}%", "Reduction",
         "PASS" if adp['page_faults'] < lru['page_faults'] else "---"],
        ["Thrashing Events", str(lru['thrashing_events']), str(adp['thrashing_events']),
         f"-{imp.get('thrashing_reduction_pct',0):.1f}%", "50%+ reduction",
         "PASS" if imp.get('thrashing_reduction_pct',0) >= 50 else "---"],
        ["Evictions", str(lru['evictions']), str(adp['evictions']),
         f"-{(1-adp['evictions']/max(1,lru['evictions']))*100:.1f}%", "Reduction",
         "PASS" if adp['evictions'] <= lru['evictions'] else "---"],
    ]
    print(tabulate(rows, headers=["Metric","LRU","Adaptive","Change","Target","Status"],
                   tablefmt="grid"))

    # ── Tiered Pre-loading + Page Clustering ──
    print(f"\n  TIERED PRE-LOADING (with Page Clustering)")
    preload_rows = [
        ["Total Preloads", adp['preloads']],
        ["Total Preload Hits", adp['preload_hits']],
        ["  Tier 1 Hits (Metadata-Warmed)", adp.get('preload_tier1_hits', 0)],
        ["  Tier 2 Hits (BG-Cached)", adp.get('preload_tier2_hits', 0)],
        ["  Tier 3 Hits (Full-RAM)", adp.get('preload_tier3_hits', 0)],
    ]
    print(tabulate(preload_rows, headers=["Metric", "Value"], tablefmt="grid"))

    # ── ZRAM Compressed Tier ──
    print(f"\n  ZRAM COMPRESSED RAM TIER")
    zram_rows = [
        ["ZRAM Stores (evicted → compressed)", adp.get('zram_stores', 0)],
        ["ZRAM Hits (decompressed → RAM)", adp.get('zram_hits', 0)],
        ["ZRAM Hit Rate", f"{adp.get('zram_hits',0)/max(1,adp.get('zram_stores',1)):.1%}"],
    ]
    print(tabulate(zram_rows, headers=["Metric", "Value"], tablefmt="grid"))

    # ── ARC Self-Tuning ──
    print(f"\n  ARC ADAPTIVE REPLACEMENT CACHE")
    arc_rows = [
        ["Ghost B1 Hits (need more recency)", adp.get('ghost_hits_b1', 0)],
        ["Ghost B2 Hits (need more frequency)", adp.get('ghost_hits_b2', 0)],
    ]
    print(tabulate(arc_rows, headers=["Metric", "Value"], tablefmt="grid"))

    # ── Energy / Battery Profiling ──
    print(f"\n  ENERGY / BATTERY PROFILING")
    lru_e = lru.get('energy_total_mj', 0)
    adp_e = adp.get('energy_total_mj', 0)
    energy_saving = imp.get('energy_reduction_pct', 0)
    energy_rows = [
        ["Total Energy (mJ)", f"{lru_e:.1f}", f"{adp_e:.1f}",
         f"-{energy_saving:.1f}%"],
        ["Storage I/O Energy (mJ)",
         f"{lru.get('energy_storage_mj',0):.1f}",
         f"{adp.get('energy_storage_mj',0):.1f}", ""],
        ["RAM Hold Energy (mJ)",
         f"{lru.get('energy_ram_hold_mj',0):.1f}",
         f"{adp.get('energy_ram_hold_mj',0):.1f}", ""],
        ["Preload Energy (mJ)", "N/A",
         f"{adp.get('energy_preload_mj',0):.1f}", ""],
        ["NPU Inference Energy (mJ)", "N/A",
         f"{adp.get('energy_inference_mj',0):.1f}", ""],
    ]
    print(tabulate(energy_rows,
                   headers=["Metric", "LRU", "Adaptive", "Saving"],
                   tablefmt="grid"))

    # ── KV Cache Management ──
    print(f"\n  KV CACHE MANAGEMENT (Token-Level Prefix Caching)")
    kv_rows = [
        ["Total GenAI Requests", kv_stats["total_requests"]],
        ["Cache Reuses", kv_stats["cache_reuses"]],
        ["Prefix Dedup Hits", kv_stats.get("prefix_dedup_hits", 0)],
        ["Quantizations (FP16->INT8)", kv_stats.get("quantizations", 0)],
        ["Offloads to Storage", kv_stats["offloads"]],
        ["Full Evictions", kv_stats["compressions"]],
        ["Total Memory Saved (MB)", f"{kv_stats['memory_saved_mb']:.1f}"],
        ["KV Energy Cost (mJ)", f"{kv_stats.get('energy_mj', 0):.1f}"],
    ]
    print(tabulate(kv_rows, headers=["Metric", "Value"], tablefmt="grid"))

    # ── Memory Pressure Comparison ──
    if pressure_results:
        print(f"\n  MEMORY PRESSURE TEST (2GB RAM)")
        pr = pressure_results
        pi = pr["improvements"]
        pressure_rows = [
            ["Cache Hit Rate",
             f"{pr['lru']['hit_rate']:.1%}", f"{pr['adaptive']['hit_rate']:.1%}",
             f"+{pi['hit_rate_improvement_pct']:.1f}%"],
            ["Avg Load Time",
             f"{pr['lru']['avg_load_time_ms']:.1f}ms",
             f"{pr['adaptive']['avg_load_time_ms']:.1f}ms",
             f"-{pi.get('load_time_reduction_pct',0):.1f}%"],
            ["Page Faults",
             str(pr['lru']['page_faults']), str(pr['adaptive']['page_faults']),
             f"-{pi.get('page_fault_reduction_pct',0):.1f}%"],
            ["Thrashing",
             str(pr['lru']['thrashing_events']), str(pr['adaptive']['thrashing_events']),
             f"-{pi.get('thrashing_reduction_pct',0):.1f}%"],
            ["Energy (mJ)",
             f"{pr['lru'].get('energy_total_mj',0):.0f}",
             f"{pr['adaptive'].get('energy_total_mj',0):.0f}",
             f"-{pi.get('energy_reduction_pct',0):.1f}%"],
        ]
        print(tabulate(pressure_rows,
                       headers=["Metric", "LRU (2GB)", "Adaptive (2GB)", "Change"],
                       tablefmt="grid"))

    # ── Overall ──
    tc = adp['thrashing_events']
    passed = sum([acc >= 0.75 or t3 >= 0.75,
                  adp['hit_rate'] >= 0.85,
                  imp.get('load_time_reduction_pct', 0) >= 10,
                  imp.get('thrashing_reduction_pct', 0) >= 50,
                  tc == 0,
                  energy_saving > 0])
    stability = '0 issues (PASS)' if tc == 0 else f'{tc} events'
    print(f"\n  KPIs Met: {passed}/6 | System Stability: {stability}")
    if energy_saving > 0:
        print(f"  Battery Saving: {energy_saving:.1f}% less energy consumption")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "data", "app_usage_logs.csv"))
    kv_df = pd.read_csv(os.path.join(base, "data", "kv_cache_workload.csv"))
    ens, dev = load_ensemble(os.path.join(base, "models"))

    # Standard simulation (4GB RAM)
    results = run_simulation(df, ens, dev, total_memory_mb=4096, label="Standard 4GB")
    # Memory pressure test (2GB RAM)
    pressure = run_simulation(df, ens, dev, total_memory_mb=2048, label="Pressure 2GB")
    # KV cache
    kv_stats = run_kv_cache_simulation(kv_df)
    # ONNX export
    export_onnx(os.path.join(base, "models"), dev)

    print_results(results, kv_stats, pressure)

    rd = os.path.join(base, "results")
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(rd, "pressure_results.json"), "w") as f:
        json.dump(pressure, f, indent=2, default=str)
    with open(os.path.join(rd, "kv_cache_results.json"), "w") as f:
        json.dump(kv_stats, f, indent=2, default=str)
