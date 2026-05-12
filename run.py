#!/usr/bin/env python3
"""
Main Runner v2 — Context-Aware Adaptive Memory Management System
Orchestrates: Data Gen -> Train -> Simulate -> Pressure Test -> ONNX -> Visualize
"""

import os, sys, time, json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def banner(text):
    print(f"\n{'='*70}\n  {text}\n{'='*70}\n")


def main():
    start = time.time()
    data_dir = os.path.join(BASE_DIR, "data")
    model_dir = os.path.join(BASE_DIR, "models")
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ── Step 1: Generate Data ──
    banner("STEP 1/7: Generating Synthetic Datasets")
    from src.data_generator import AppUsageDataGenerator, generate_kv_cache_workload, APPS
    gen = AppUsageDataGenerator(num_users=50, seed=42)
    df = gen.generate(days=30)
    df.to_csv(os.path.join(data_dir, "app_usage_logs.csv"), index=False)
    print(f"  App usage logs: {len(df):,} records, {df['user_id'].nunique()} users")
    with open(os.path.join(data_dir, "app_metadata.json"), "w") as f:
        json.dump(APPS, f, indent=2, default=str)
    kv_df = generate_kv_cache_workload(500, 42)
    kv_df.to_csv(os.path.join(data_dir, "kv_cache_workload.csv"), index=False)
    print(f"  KV cache workload: {len(kv_df)} requests")

    # ── Step 2: Train Predictor ──
    banner("STEP 2/7: Training Ensemble Predictor (LSTM + Markov)")
    from src.predictor import train_model
    lstm_model, markov_model, history = train_model(df, epochs=40, batch_size=256,
                                                     model_dir=model_dir)

    # ── Step 3: Standard Simulation (4GB) ──
    banner("STEP 3/7: Running Memory Simulation (4GB RAM)")
    from simulate import (load_ensemble, run_simulation, run_kv_cache_simulation,
                          print_results, export_onnx)
    import pandas as pd

    ensemble, device = load_ensemble(model_dir)
    results = run_simulation(df, ensemble, device, total_memory_mb=4096,
                             label="Standard 4GB")

    # ── Step 4: Memory Pressure Test (2GB) ──
    banner("STEP 4/7: Memory Pressure Test (2GB RAM)")
    pressure = run_simulation(df, ensemble, device, total_memory_mb=2048,
                              label="Pressure 2GB")

    # ── Step 5: KV Cache Simulation ──
    banner("STEP 5/7: KV Cache Simulation (Token-Level Prefix Caching)")
    kv_stats = run_kv_cache_simulation(kv_df)

    # ── Step 6: ONNX Export (Sim-to-Real Bridge) ──
    banner("STEP 6/7: ONNX Model Export (Edge Deployment)")
    onnx_path, onnx_size = export_onnx(model_dir, device)

    # ── Step 7: Results & Viz ──
    banner("STEP 7/7: Results & Visualizations")
    with open(os.path.join(results_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(results_dir, "pressure_results.json"), "w") as f:
        json.dump(pressure, f, indent=2, default=str)
    with open(os.path.join(results_dir, "kv_cache_results.json"), "w") as f:
        json.dump(kv_stats, f, indent=2, default=str)

    print_results(results, kv_stats, pressure)

    from src.visualize import (plot_training_curves, plot_kpi_comparison,
                               plot_prediction_analysis, plot_pressure_comparison)
    hist_path = os.path.join(model_dir, "training_history.json")
    res_path = os.path.join(results_dir, "benchmark_results.json")
    pres_path = os.path.join(results_dir, "pressure_results.json")
    if os.path.exists(hist_path):
        plot_training_curves(hist_path, results_dir)
    if os.path.exists(res_path):
        plot_kpi_comparison(res_path, results_dir)
        plot_prediction_analysis(res_path, results_dir)
    if os.path.exists(pres_path):
        plot_pressure_comparison(res_path, pres_path, results_dir)

    elapsed = time.time() - start
    banner(f"COMPLETE - Total time: {elapsed:.1f}s")
    print(f"  Data:    {data_dir}/")
    print(f"  Models:  {model_dir}/")
    print(f"  Results: {results_dir}/")
    if onnx_path:
        print(f"  ONNX:    {onnx_path} ({onnx_size:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
