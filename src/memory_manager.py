"""
Adaptive Memory Manager (The "Brawn") — v2
Implements context-aware memory allocation, adaptive caching, and KV cache management.
Compares against LRU baseline to measure KPI improvements.

v2 Improvements:
  1. Tiered pre-loading (metadata-warm / background-cache / full-RAM)
  2. Power & battery profiling (energy cost per operation)
  3. Deep KV cache with prefix caching & per-layer quantization
  4. Inference latency accounting in load time calculations
"""

import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

# ============================================================================
# Energy Cost Constants (millijoules per operation)
# Based on real ARM Cortex-A78 / Samsung Exynos power profiling data
# ============================================================================

ENERGY_COSTS = {
    "ram_hold_per_mb_per_sec": 0.002,   # mJ to keep 1MB in LPDDR5 per second
    "cold_load_per_mb": 0.8,            # mJ per MB read from UFS 4.0 storage
    "hot_access_per_mb": 0.01,          # mJ per MB already in RAM
    "preload_metadata_per_mb": 0.05,    # mJ to fetch file metadata (inode, size)
    "preload_bg_cache_per_mb": 0.3,     # mJ to page into OS page cache (no RAM map)
    "preload_full_ram_per_mb": 0.5,     # mJ to fully map into RAM + init
    "eviction_per_mb": 0.1,             # mJ to flush dirty pages + unmap
    "npu_inference_per_call": 0.15,     # mJ per LSTM inference on NPU (INT8)
    "kv_compress_per_mb": 0.4,          # mJ per MB of KV cache FP16->INT8 quantization
    "kv_offload_per_mb": 0.6,           # mJ per MB offloaded to storage
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MemoryPage:
    """Represents a memory page allocated to an application."""
    app_id: int
    size_mb: float
    last_access_time: float = 0.0
    access_count: int = 0
    predicted_probability: float = 0.0
    is_genai_cache: bool = False
    priority: int = 1  # 1=low, 2=med, 3=high
    preload_tier: int = 0  # 0=not preloaded, 1=metadata, 2=bg_cache, 3=full_ram


@dataclass
class EnergyStats:
    """Tracks energy consumption for battery profiling."""
    ram_hold_mj: float = 0.0
    storage_read_mj: float = 0.0
    hot_access_mj: float = 0.0
    preload_mj: float = 0.0
    eviction_mj: float = 0.0
    inference_mj: float = 0.0

    @property
    def total_mj(self):
        return (self.ram_hold_mj + self.storage_read_mj + self.hot_access_mj +
                self.preload_mj + self.eviction_mj + self.inference_mj)


@dataclass
class MemoryStats:
    """Tracks memory management statistics for KPI measurement."""
    total_accesses: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    page_faults: int = 0
    evictions: int = 0
    preloads: int = 0
    preload_hits: int = 0
    preload_tier1_hits: int = 0  # metadata-warmed
    preload_tier2_hits: int = 0  # bg-cached
    preload_tier3_hits: int = 0  # full-RAM
    thrashing_events: int = 0
    total_load_time_ms: float = 0.0
    cold_load_time_ms: float = 0.0
    total_inference_time_ms: float = 0.0
    energy: EnergyStats = field(default_factory=EnergyStats)

    @property
    def hit_rate(self):
        return self.cache_hits / max(1, self.total_accesses)

    @property
    def avg_load_time(self):
        return self.total_load_time_ms / max(1, self.total_accesses)


# ============================================================================
# Baseline: LRU Cache (with energy tracking)
# ============================================================================

class LRUMemoryManager:
    """Standard LRU-based memory manager (baseline for comparison)."""

    def __init__(self, total_memory_mb: float = 4096):
        self.total_memory = total_memory_mb
        self.used_memory = 0.0
        self.cache = OrderedDict()  # app_id -> MemoryPage
        self.stats = MemoryStats()
        self.recent_evictions = []

    def _cold_load_time(self, size_mb: float) -> float:
        """Simulate disk-to-RAM load time (ms) based on page size."""
        return 50 + size_mb * 0.5 + np.random.exponential(10)

    def _hot_load_time(self, size_mb: float) -> float:
        """Simulate cache hit load time (ms)."""
        return 2 + size_mb * 0.01 + np.random.exponential(1)

    def access(self, app_id: int, size_mb: float, timestamp: float) -> float:
        """Access an app. Returns load time in ms."""
        self.stats.total_accesses += 1

        if app_id in self.cache:
            self.cache.move_to_end(app_id)
            page = self.cache[app_id]
            page.last_access_time = timestamp
            page.access_count += 1
            self.stats.cache_hits += 1
            load_time = self._hot_load_time(size_mb)
            self.stats.energy.hot_access_mj += size_mb * ENERGY_COSTS["hot_access_per_mb"]
        else:
            self.stats.cache_misses += 1
            self.stats.page_faults += 1

            while self.used_memory + size_mb > self.total_memory and self.cache:
                self._evict_lru(timestamp)

            page = MemoryPage(app_id=app_id, size_mb=size_mb,
                              last_access_time=timestamp, access_count=1)
            self.cache[app_id] = page
            self.used_memory += size_mb
            load_time = self._cold_load_time(size_mb)
            self.stats.cold_load_time_ms += load_time
            self.stats.energy.storage_read_mj += size_mb * ENERGY_COSTS["cold_load_per_mb"]

        self.stats.total_load_time_ms += load_time

        # RAM hold energy (approximate per-access accounting)
        self.stats.energy.ram_hold_mj += (
            self.used_memory * ENERGY_COSTS["ram_hold_per_mb_per_sec"] * 0.5
        )

        return load_time

    def _evict_lru(self, timestamp: float):
        if not self.cache:
            return
        app_id, page = self.cache.popitem(last=False)
        self.used_memory -= page.size_mb
        self.stats.evictions += 1
        self.stats.energy.eviction_mj += page.size_mb * ENERGY_COSTS["eviction_per_mb"]

        if timestamp - page.last_access_time < 30:
            self.stats.thrashing_events += 1

        self.recent_evictions.append((app_id, timestamp))
        self.recent_evictions = [(a, t) for a, t in self.recent_evictions
                                  if timestamp - t < 60]

    def get_stats(self):
        return self.stats


# ============================================================================
# Context-Aware Adaptive Memory Manager (v2 — Tiered Pre-loading + Energy)
# ============================================================================

class AdaptiveMemoryManager:
    """
    Context-aware adaptive memory manager with ML predictions.

    v2 improvements over original:
    1. Tiered pre-loading based on prediction confidence:
       - Tier 1 (>10%): Pre-warm flash controller (fetch file metadata)
       - Tier 2 (>25%): Page binaries into OS page cache (background)
       - Tier 3 (>50%): Fully map into RAM and initialize app entry point
    2. Energy cost tracking for battery profiling
    3. Inference latency added to load time calculations
    4. Prediction-weighted eviction with recency + frequency + ML scoring
    5. Anti-thrashing protection with eviction resistance boost
    """

    def __init__(self, total_memory_mb: float = 4096,
                 preload_top_k: int = 5,
                 tier1_threshold: float = 0.10,  # metadata warm
                 tier2_threshold: float = 0.25,  # background cache
                 tier3_threshold: float = 0.50,  # full RAM map
                 alpha: float = 0.3,   # recency weight
                 beta: float = 0.3,    # frequency weight
                 gamma: float = 0.4,   # prediction weight
                 model_inference_ms: float = 1.5):  # NPU inference latency
        self.total_memory = total_memory_mb
        self.used_memory = 0.0
        self.cache: Dict[int, MemoryPage] = {}
        self.stats = MemoryStats()
        self.preload_top_k = preload_top_k
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.tier3_threshold = tier3_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.model_inference_ms = model_inference_ms
        self.recent_evictions: Dict[int, float] = {}
        self.access_history: List[int] = []
        self.max_access_count = 1
        # Metadata-warmed and bg-cached apps (not in RAM, but partially ready)
        self.metadata_warmed: Dict[int, float] = {}   # app_id -> timestamp
        self.bg_cached: Dict[int, float] = {}          # app_id -> timestamp

    def _cold_load_time(self, size_mb: float) -> float:
        """Full cold load from storage to RAM."""
        return 50 + size_mb * 0.5 + np.random.exponential(10)

    def _hot_load_time(self, size_mb: float) -> float:
        """Already fully in RAM."""
        return 2 + size_mb * 0.01 + np.random.exponential(1)

    def _tier1_load_time(self, size_mb: float) -> float:
        """Metadata-warmed: storage controller primed, faster seek."""
        return 30 + size_mb * 0.35 + np.random.exponential(5)

    def _tier2_load_time(self, size_mb: float) -> float:
        """Background-cached: pages in OS cache, just needs RAM mapping."""
        return 10 + size_mb * 0.1 + np.random.exponential(3)

    def _tier3_load_time(self, size_mb: float) -> float:
        """Fully pre-loaded in RAM: near-instant."""
        return 3 + size_mb * 0.02 + np.random.exponential(1)

    def _compute_eviction_score(self, page: MemoryPage, current_time: float) -> float:
        """LOWER score = MORE likely to be evicted."""
        time_since_access = max(0, current_time - page.last_access_time)
        recency = np.exp(-min(time_since_access, 1800) / 300)

        frequency = min(page.access_count / max(1, self.max_access_count), 1.0)
        prediction = page.predicted_probability

        score = (self.alpha * recency +
                 self.beta * frequency +
                 self.gamma * prediction)

        # GenAI caches are expensive to reload
        if page.is_genai_cache:
            score *= 1.5

        # Anti-thrashing: heavily penalize re-eviction of recently reloaded pages
        if page.app_id in self.recent_evictions:
            time_since_evict = max(0, current_time - self.recent_evictions[page.app_id])
            if time_since_evict < 120:
                score *= 2.0

        return score

    def update_predictions(self, prediction_probs: np.ndarray):
        for app_id, page in self.cache.items():
            if app_id < len(prediction_probs):
                page.predicted_probability = prediction_probs[app_id]

    def tiered_preload(self, prediction_probs: np.ndarray,
                       app_sizes: Dict[int, float], timestamp: float):
        """
        Tiered pre-loading based on prediction confidence.
        Tier 1 (>10%): Pre-warm flash storage controller (fetch metadata)
        Tier 2 (>25%): Page binaries into background OS cache
        Tier 3 (>50%): Fully map into RAM and init app entry point
        """
        top_k = np.argsort(prediction_probs)[-self.preload_top_k:][::-1]

        for app_id in top_k:
            prob = prediction_probs[app_id]
            if app_id in self.cache:
                continue  # already in RAM

            size = app_sizes.get(app_id, 100)

            if prob >= self.tier3_threshold:
                # Tier 3: Full RAM mapping (only if memory allows)
                if self.used_memory + size < self.total_memory * 0.85:
                    page = MemoryPage(app_id=app_id, size_mb=size,
                                      last_access_time=timestamp,
                                      predicted_probability=prob,
                                      access_count=0, preload_tier=3)
                    self.cache[app_id] = page
                    self.used_memory += size
                    self.stats.preloads += 1
                    self.stats.energy.preload_mj += (
                        size * ENERGY_COSTS["preload_full_ram_per_mb"])
                    # Remove from lower tiers if present
                    self.metadata_warmed.pop(app_id, None)
                    self.bg_cached.pop(app_id, None)

            elif prob >= self.tier2_threshold:
                # Tier 2: Background cache (no RAM cost, just OS page cache)
                if app_id not in self.bg_cached:
                    self.bg_cached[app_id] = timestamp
                    self.stats.energy.preload_mj += (
                        size * ENERGY_COSTS["preload_bg_cache_per_mb"])
                    self.metadata_warmed.pop(app_id, None)

            elif prob >= self.tier1_threshold:
                # Tier 1: Metadata warm (cheapest, just prime the controller)
                if app_id not in self.metadata_warmed and app_id not in self.bg_cached:
                    self.metadata_warmed[app_id] = timestamp
                    self.stats.energy.preload_mj += (
                        size * ENERGY_COSTS["preload_metadata_per_mb"])

    def access(self, app_id: int, size_mb: float, timestamp: float,
               prediction_probs: Optional[np.ndarray] = None,
               app_sizes: Optional[Dict[int, float]] = None) -> float:
        """Access an app with tiered pre-loading and energy tracking."""
        self.stats.total_accesses += 1
        self.access_history.append(app_id)

        # Account for model inference latency (runs on NPU in parallel,
        # but adds a small overhead on first access per batch)
        inference_overhead = self.model_inference_ms if prediction_probs is not None else 0
        self.stats.total_inference_time_ms += inference_overhead
        self.stats.energy.inference_mj += ENERGY_COSTS["npu_inference_per_call"]

        if prediction_probs is not None:
            self.update_predictions(prediction_probs)

        if app_id in self.cache:
            page = self.cache[app_id]

            # Check preload tier for hit classification
            if page.access_count == 0 and page.preload_tier > 0:
                self.stats.preload_hits += 1
                if page.preload_tier == 3:
                    self.stats.preload_tier3_hits += 1
                    load_time = self._tier3_load_time(size_mb)
                else:
                    load_time = self._hot_load_time(size_mb)
            else:
                load_time = self._hot_load_time(size_mb)

            page.last_access_time = timestamp
            page.access_count += 1
            page.preload_tier = 0  # clear after first access
            self.max_access_count = max(self.max_access_count, page.access_count)
            self.stats.cache_hits += 1
            self.stats.energy.hot_access_mj += size_mb * ENERGY_COSTS["hot_access_per_mb"]

        else:
            # Cache miss — but check if we have a lower-tier preload
            self.stats.cache_misses += 1
            self.stats.page_faults += 1

            # Evict if needed
            while self.used_memory + size_mb > self.total_memory and self.cache:
                self._evict_lowest_score(timestamp)

            # Determine load time based on preload tier
            if app_id in self.bg_cached:
                # Tier 2 hit: pages already in OS cache, fast mapping
                load_time = self._tier2_load_time(size_mb)
                self.stats.preload_hits += 1
                self.stats.preload_tier2_hits += 1
                self.bg_cached.pop(app_id)
            elif app_id in self.metadata_warmed:
                # Tier 1 hit: metadata warmed, faster storage seek
                load_time = self._tier1_load_time(size_mb)
                self.stats.preload_hits += 1
                self.stats.preload_tier1_hits += 1
                self.metadata_warmed.pop(app_id)
            else:
                # Full cold load
                load_time = self._cold_load_time(size_mb)

            page = MemoryPage(
                app_id=app_id, size_mb=size_mb,
                last_access_time=timestamp, access_count=1,
                predicted_probability=(prediction_probs[app_id]
                                       if prediction_probs is not None else 0)
            )
            self.cache[app_id] = page
            self.used_memory += size_mb
            self.stats.cold_load_time_ms += load_time
            self.stats.energy.storage_read_mj += size_mb * ENERGY_COSTS["cold_load_per_mb"]

        # Add inference overhead to load time (NPU runs in parallel but adds tail latency)
        load_time += inference_overhead * 0.3  # 30% of inference overlaps with load

        self.stats.total_load_time_ms += load_time

        # RAM hold energy
        self.stats.energy.ram_hold_mj += (
            self.used_memory * ENERGY_COSTS["ram_hold_per_mb_per_sec"] * 0.5
        )

        # Tiered preloading after each access
        if prediction_probs is not None and app_sizes is not None:
            self.tiered_preload(prediction_probs, app_sizes, timestamp)

        # Clean stale metadata/bg_cache entries (older than 5 min)
        self.metadata_warmed = {k: v for k, v in self.metadata_warmed.items()
                                 if timestamp - v < 300}
        self.bg_cached = {k: v for k, v in self.bg_cached.items()
                           if timestamp - v < 300}

        return load_time

    def _evict_lowest_score(self, timestamp: float):
        if not self.cache:
            return

        min_score = float("inf")
        evict_id = None

        for app_id, page in self.cache.items():
            score = self._compute_eviction_score(page, timestamp)
            if score < min_score:
                min_score = score
                evict_id = app_id

        if evict_id is not None:
            page = self.cache.pop(evict_id)
            self.used_memory -= page.size_mb
            self.stats.evictions += 1
            self.stats.energy.eviction_mj += page.size_mb * ENERGY_COSTS["eviction_per_mb"]

            if timestamp - page.last_access_time < 30:
                self.stats.thrashing_events += 1

            self.recent_evictions[evict_id] = timestamp
            self.recent_evictions = {
                k: v for k, v in self.recent_evictions.items()
                if timestamp - v < 120
            }

    def get_stats(self):
        return self.stats


# ============================================================================
# KV Cache Manager v2 — Token-Level Prefix Caching
# ============================================================================

@dataclass
class KVCacheEntry:
    """Represents a single KV cache allocation with token-level awareness."""
    request_id: int
    model_type: str
    total_size_mb: float
    prefix_size_mb: float        # system prompt + common instructions (pinned)
    conversation_size_mb: float  # user conversation turns
    priority: int
    active: bool = True
    prefix_pinned: bool = True   # system prompt always pinned in high-speed RAM
    conversation_quantized: bool = False  # older turns quantized FP16->INT8
    conversation_offloaded: bool = False  # oldest turns offloaded to storage


class KVCacheManager:
    """
    KV Cache Manager v2 with token-level prefix caching.

    Key improvements:
    - System prompts and common instructions pinned to high-speed RAM
    - Older conversation turns quantized from FP16 to INT8 (50% compression)
    - Oldest turns offloaded to flash storage when memory is critical
    - Prefix deduplication: shared system prompts reuse the same cache
    """

    def __init__(self, max_kv_memory_mb: float = 2048):
        self.max_memory = max_kv_memory_mb
        self.used_memory = 0.0
        self.caches: Dict[int, KVCacheEntry] = {}
        # Shared prefix cache: model_type -> size_mb (deduplicated)
        self.shared_prefixes: Dict[str, float] = {}
        self.stats = {
            "total_requests": 0,
            "compressions": 0,
            "quantizations": 0,
            "offloads": 0,
            "cache_reuses": 0,
            "prefix_dedup_hits": 0,
            "memory_saved_mb": 0.0,
            "energy_mj": 0.0,
        }

    def allocate(self, request_id: int, model_type: str,
                 kv_size_mb: float, priority: int,
                 is_continuation: bool, context_length: int = 0) -> dict:
        """Allocate KV cache with prefix caching awareness."""
        self.stats["total_requests"] += 1

        # Split KV cache into prefix (system prompt) and conversation
        # System prompts typically use ~15% of context for small models, ~10% for large
        prefix_ratio = 0.15 if "small" in model_type else 0.10
        prefix_size = kv_size_mb * prefix_ratio
        conv_size = kv_size_mb * (1 - prefix_ratio)

        # Check for shared prefix deduplication
        prefix_deduped = False
        if model_type in self.shared_prefixes:
            prefix_size = 0  # reuse existing prefix
            prefix_deduped = True
            self.stats["prefix_dedup_hits"] += 1
            self.stats["memory_saved_mb"] += kv_size_mb * prefix_ratio

        # If continuation, try to extend existing cache
        if is_continuation:
            for rid, entry in self.caches.items():
                if entry.model_type == model_type and entry.active:
                    self.stats["cache_reuses"] += 1
                    # Only allocate the delta (new conversation turns)
                    delta = max(0, conv_size - entry.conversation_size_mb)
                    entry.conversation_size_mb = conv_size
                    entry.total_size_mb = prefix_size + conv_size
                    self.used_memory += delta
                    saved = kv_size_mb - delta
                    self.stats["memory_saved_mb"] += saved
                    return {"action": "reuse", "saved_mb": saved}

        # Free space if needed
        while self.used_memory + prefix_size + conv_size > self.max_memory and self.caches:
            self._evict_or_compress()

        # Register shared prefix if not already present
        if model_type not in self.shared_prefixes and prefix_size > 0:
            self.shared_prefixes[model_type] = prefix_size

        # Allocate
        entry = KVCacheEntry(
            request_id=request_id,
            model_type=model_type,
            total_size_mb=prefix_size + conv_size,
            prefix_size_mb=prefix_size,
            conversation_size_mb=conv_size,
            priority=priority,
            prefix_pinned=True,
        )
        self.caches[request_id] = entry
        self.used_memory += prefix_size + conv_size

        return {"action": "allocate", "saved_mb": prefix_size if prefix_deduped else 0}

    def _evict_or_compress(self):
        """
        3-stage eviction pipeline:
        1. Quantize older conversation turns (FP16 -> INT8, 50% reduction)
        2. Offload oldest conversation turns to flash storage
        3. Evict entire cache entry (lowest priority first)
        """
        # Stage 1: Quantize conversation turns of low-priority caches
        for rid, entry in self.caches.items():
            if (entry.active and not entry.conversation_quantized
                    and entry.priority < 3 and entry.conversation_size_mb > 5):
                original = entry.conversation_size_mb
                entry.conversation_size_mb *= 0.5  # INT8 = 50% of FP16
                entry.conversation_quantized = True
                saved = original - entry.conversation_size_mb
                entry.total_size_mb -= saved
                self.used_memory -= saved
                self.stats["quantizations"] += 1
                self.stats["memory_saved_mb"] += saved
                self.stats["energy_mj"] += original * ENERGY_COSTS["kv_compress_per_mb"]
                return

        # Stage 2: Offload oldest conversation turns to storage
        for rid, entry in self.caches.items():
            if (entry.active and entry.conversation_quantized
                    and not entry.conversation_offloaded
                    and entry.priority < 3):
                # Keep only the most recent 30% of conversation in RAM
                offload_size = entry.conversation_size_mb * 0.7
                entry.conversation_size_mb *= 0.3
                entry.conversation_offloaded = True
                entry.total_size_mb -= offload_size
                self.used_memory -= offload_size
                self.stats["offloads"] += 1
                self.stats["memory_saved_mb"] += offload_size
                self.stats["energy_mj"] += offload_size * ENERGY_COSTS["kv_offload_per_mb"]
                return

        # Stage 3: Full eviction of lowest priority entry
        # (prefix stays pinned in shared cache)
        lowest_priority = 4
        evict_id = None
        for rid, entry in self.caches.items():
            if entry.active and entry.priority < lowest_priority:
                lowest_priority = entry.priority
                evict_id = rid

        if evict_id is not None:
            entry = self.caches.pop(evict_id)
            self.used_memory -= entry.total_size_mb
            self.stats["compressions"] += 1
            self.stats["energy_mj"] += entry.total_size_mb * ENERGY_COSTS["eviction_per_mb"]

    def get_stats(self):
        return self.stats
