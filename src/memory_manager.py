"""
Adaptive Memory Manager (The "Brawn")
Implements context-aware memory allocation, adaptive caching, and KV cache management.
Compares against LRU baseline to measure KPI improvements.
"""

import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

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
    thrashing_events: int = 0  # rapid evict-then-reload cycles
    total_load_time_ms: float = 0.0
    cold_load_time_ms: float = 0.0

    @property
    def hit_rate(self):
        return self.cache_hits / max(1, self.total_accesses)

    @property
    def avg_load_time(self):
        return self.total_load_time_ms / max(1, self.total_accesses)


# ============================================================================
# Baseline: LRU Cache
# ============================================================================

class LRUMemoryManager:
    """Standard LRU-based memory manager (baseline for comparison)."""

    def __init__(self, total_memory_mb: float = 4096):
        self.total_memory = total_memory_mb
        self.used_memory = 0.0
        self.cache = OrderedDict()  # app_id -> MemoryPage
        self.stats = MemoryStats()
        self.recent_evictions = []  # track for thrashing detection

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
            # Cache hit - move to end (most recent)
            self.cache.move_to_end(app_id)
            page = self.cache[app_id]
            page.last_access_time = timestamp
            page.access_count += 1
            self.stats.cache_hits += 1
            load_time = self._hot_load_time(size_mb)
        else:
            # Cache miss - need to load from disk
            self.stats.cache_misses += 1
            self.stats.page_faults += 1

            # Evict if necessary
            while self.used_memory + size_mb > self.total_memory and self.cache:
                self._evict_lru(timestamp)

            # Load new page
            page = MemoryPage(app_id=app_id, size_mb=size_mb,
                              last_access_time=timestamp, access_count=1)
            self.cache[app_id] = page
            self.used_memory += size_mb
            load_time = self._cold_load_time(size_mb)
            self.stats.cold_load_time_ms += load_time

        self.stats.total_load_time_ms += load_time
        return load_time

    def _evict_lru(self, timestamp: float):
        """Evict the least recently used page."""
        if not self.cache:
            return
        app_id, page = self.cache.popitem(last=False)  # pop oldest
        self.used_memory -= page.size_mb
        self.stats.evictions += 1

        # Thrashing detection: if we evict something we loaded recently
        if timestamp - page.last_access_time < 30:  # within 30 seconds
            self.stats.thrashing_events += 1

        self.recent_evictions.append((app_id, timestamp))
        # Keep only recent evictions
        self.recent_evictions = [(a, t) for a, t in self.recent_evictions
                                  if timestamp - t < 60]

    def get_stats(self):
        return self.stats


# ============================================================================
# Context-Aware Adaptive Memory Manager
# ============================================================================

class AdaptiveMemoryManager:
    """
    Context-aware adaptive memory manager that uses ML predictions
    for intelligent caching, pre-loading, and eviction.
    
    Key innovations over LRU:
    1. Prediction-weighted eviction (evict lowest predicted probability)
    2. Proactive pre-loading based on predicted next apps
    3. Frequency + recency + prediction combined scoring
    4. GenAI KV-cache aware memory management
    5. Anti-thrashing protection
    """

    def __init__(self, total_memory_mb: float = 4096,
                 preload_threshold: float = 0.15,
                 preload_top_k: int = 3,
                 alpha: float = 0.3,  # weight for recency
                 beta: float = 0.3,   # weight for frequency
                 gamma: float = 0.4): # weight for prediction
        self.total_memory = total_memory_mb
        self.used_memory = 0.0
        self.cache: Dict[int, MemoryPage] = {}
        self.stats = MemoryStats()
        self.preload_threshold = preload_threshold
        self.preload_top_k = preload_top_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.recent_evictions: Dict[int, float] = {}  # app_id -> eviction_time
        self.access_history: List[int] = []
        self.max_access_count = 1  # for normalization

    def _cold_load_time(self, size_mb: float) -> float:
        return 50 + size_mb * 0.5 + np.random.exponential(10)

    def _hot_load_time(self, size_mb: float) -> float:
        return 2 + size_mb * 0.01 + np.random.exponential(1)

    def _preloaded_load_time(self, size_mb: float) -> float:
        """Pre-loaded apps are warm but not fully hot."""
        return 5 + size_mb * 0.05 + np.random.exponential(2)

    def _compute_eviction_score(self, page: MemoryPage, current_time: float) -> float:
        """
        Compute eviction priority score. LOWER score = MORE likely to be evicted.
        Combines recency, frequency, and predicted probability.
        """
        # Recency score: how recently was it accessed (normalized to 0-1)
        time_since_access = max(0, current_time - page.last_access_time)
        # Clamp to avoid overflow; anything older than 30 min has ~0 recency
        recency = np.exp(-min(time_since_access, 1800) / 300)  # decay over 5 minutes

        # Frequency score: normalized access count
        frequency = min(page.access_count / max(1, self.max_access_count), 1.0)

        # Prediction score: ML-predicted probability of future use
        prediction = page.predicted_probability

        # Combined score
        score = (self.alpha * recency +
                 self.beta * frequency +
                 self.gamma * prediction)

        # Boost for GenAI caches (expensive to reload)
        if page.is_genai_cache:
            score *= 1.5

        # Anti-thrashing: boost recently evicted-then-reloaded apps
        if page.app_id in self.recent_evictions:
            time_since_evict = current_time - self.recent_evictions[page.app_id]
            if time_since_evict < 120:
                score *= 2.0  # strongly discourage re-eviction

        return score

    def update_predictions(self, prediction_probs: np.ndarray):
        """
        Update predicted probabilities for all apps based on ML model output.
        Called after each app access with the model's probability distribution.
        """
        for app_id, page in self.cache.items():
            if app_id < len(prediction_probs):
                page.predicted_probability = prediction_probs[app_id]

    def preload(self, prediction_probs: np.ndarray, app_sizes: Dict[int, float],
                timestamp: float):
        """
        Proactively load top-K predicted apps into memory.
        Only preloads if probability exceeds threshold and memory is available.
        """
        top_k_apps = np.argsort(prediction_probs)[-self.preload_top_k:][::-1]

        for app_id in top_k_apps:
            prob = prediction_probs[app_id]
            if prob < self.preload_threshold:
                continue
            if app_id in self.cache:
                continue  # already loaded

            size = app_sizes.get(app_id, 100)

            # Only preload if we have enough spare memory (keep 20% buffer)
            if self.used_memory + size < self.total_memory * 0.8:
                page = MemoryPage(
                    app_id=app_id, size_mb=size,
                    last_access_time=timestamp,
                    predicted_probability=prob,
                    access_count=0
                )
                self.cache[app_id] = page
                self.used_memory += size
                self.stats.preloads += 1

    def access(self, app_id: int, size_mb: float, timestamp: float,
               prediction_probs: Optional[np.ndarray] = None,
               app_sizes: Optional[Dict[int, float]] = None) -> float:
        """
        Access an app with context-aware memory management.
        Returns load time in ms.
        """
        self.stats.total_accesses += 1
        self.access_history.append(app_id)

        # Update predictions if provided
        if prediction_probs is not None:
            self.update_predictions(prediction_probs)

        if app_id in self.cache:
            page = self.cache[app_id]

            # Check if this was a preloaded hit
            if page.access_count == 0:
                self.stats.preload_hits += 1
                load_time = self._preloaded_load_time(size_mb)
            else:
                load_time = self._hot_load_time(size_mb)

            page.last_access_time = timestamp
            page.access_count += 1
            self.max_access_count = max(self.max_access_count, page.access_count)
            self.stats.cache_hits += 1
        else:
            # Cache miss
            self.stats.cache_misses += 1
            self.stats.page_faults += 1

            # Evict if necessary using intelligent scoring
            while self.used_memory + size_mb > self.total_memory and self.cache:
                self._evict_lowest_score(timestamp)

            page = MemoryPage(
                app_id=app_id, size_mb=size_mb,
                last_access_time=timestamp, access_count=1,
                predicted_probability=(prediction_probs[app_id]
                                       if prediction_probs is not None else 0)
            )
            self.cache[app_id] = page
            self.used_memory += size_mb
            load_time = self._cold_load_time(size_mb)
            self.stats.cold_load_time_ms += load_time

        self.stats.total_load_time_ms += load_time

        # Proactive preloading
        if prediction_probs is not None and app_sizes is not None:
            self.preload(prediction_probs, app_sizes, timestamp)

        return load_time

    def _evict_lowest_score(self, timestamp: float):
        """Evict the page with the lowest retention score."""
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

            # Thrashing detection
            if timestamp - page.last_access_time < 30:
                self.stats.thrashing_events += 1

            self.recent_evictions[evict_id] = timestamp
            # Clean old eviction records
            self.recent_evictions = {
                k: v for k, v in self.recent_evictions.items()
                if timestamp - v < 120
            }

    def get_stats(self):
        return self.stats


# ============================================================================
# KV Cache Manager for GenAI Workloads
# ============================================================================

class KVCacheManager:
    """
    Manages Key-Value caches for on-device GenAI models.
    Implements intelligent compression and offloading of KV cache pages.
    """

    def __init__(self, max_kv_memory_mb: float = 2048):
        self.max_memory = max_kv_memory_mb
        self.used_memory = 0.0
        self.caches: Dict[int, dict] = {}  # request_id -> cache info
        self.stats = {"total_requests": 0, "compressions": 0,
                      "offloads": 0, "cache_reuses": 0,
                      "memory_saved_mb": 0.0}

    def allocate(self, request_id: int, model_type: str,
                 kv_size_mb: float, priority: int,
                 is_continuation: bool) -> dict:
        """Allocate KV cache for a GenAI request."""
        self.stats["total_requests"] += 1

        # If continuation, try to reuse existing cache
        if is_continuation:
            for rid, info in self.caches.items():
                if info["model_type"] == model_type and info["active"]:
                    self.stats["cache_reuses"] += 1
                    info["size_mb"] = kv_size_mb  # update size
                    return {"action": "reuse", "saved_mb": kv_size_mb * 0.6}

        # Check if we need to free space
        while self.used_memory + kv_size_mb > self.max_memory and self.caches:
            self._evict_or_compress()

        # Allocate
        self.caches[request_id] = {
            "model_type": model_type,
            "size_mb": kv_size_mb,
            "priority": priority,
            "active": True,
            "compressed": False,
        }
        self.used_memory += kv_size_mb

        return {"action": "allocate", "saved_mb": 0}

    def _evict_or_compress(self):
        """Compress low-priority caches first, then evict."""
        # First try compression
        for rid, info in self.caches.items():
            if info["active"] and not info["compressed"] and info["priority"] < 3:
                original = info["size_mb"]
                info["size_mb"] *= 0.4  # 60% compression
                info["compressed"] = True
                saved = original - info["size_mb"]
                self.used_memory -= saved
                self.stats["compressions"] += 1
                self.stats["memory_saved_mb"] += saved
                return

        # Then evict lowest priority
        lowest_priority = 4
        evict_id = None
        for rid, info in self.caches.items():
            if info["active"] and info["priority"] < lowest_priority:
                lowest_priority = info["priority"]
                evict_id = rid

        if evict_id is not None:
            info = self.caches.pop(evict_id)
            self.used_memory -= info["size_mb"]
            self.stats["offloads"] += 1

    def get_stats(self):
        return self.stats
