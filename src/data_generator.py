"""
Synthetic Android App Usage Data Generator
Generates realistic smartphone app usage sequences with temporal patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

APPS = {
    0: {"name": "Phone", "category": "communication", "memory_mb": 80, "is_genai": False},
    1: {"name": "Messages", "category": "communication", "memory_mb": 120, "is_genai": False},
    2: {"name": "Chrome", "category": "browser", "memory_mb": 350, "is_genai": False},
    3: {"name": "Gmail", "category": "productivity", "memory_mb": 150, "is_genai": False},
    4: {"name": "Calendar", "category": "productivity", "memory_mb": 90, "is_genai": False},
    5: {"name": "Maps", "category": "navigation", "memory_mb": 280, "is_genai": False},
    6: {"name": "YouTube", "category": "entertainment", "memory_mb": 300, "is_genai": False},
    7: {"name": "Instagram", "category": "social", "memory_mb": 250, "is_genai": False},
    8: {"name": "Twitter_X", "category": "social", "memory_mb": 200, "is_genai": False},
    9: {"name": "WhatsApp", "category": "communication", "memory_mb": 180, "is_genai": False},
    10: {"name": "Spotify", "category": "music", "memory_mb": 160, "is_genai": False},
    11: {"name": "Camera", "category": "utility", "memory_mb": 200, "is_genai": False},
    12: {"name": "Gallery", "category": "utility", "memory_mb": 150, "is_genai": False},
    13: {"name": "Settings", "category": "system", "memory_mb": 60, "is_genai": False},
    14: {"name": "Files", "category": "utility", "memory_mb": 70, "is_genai": False},
    15: {"name": "Notes", "category": "productivity", "memory_mb": 80, "is_genai": False},
    16: {"name": "Clock_Alarm", "category": "utility", "memory_mb": 40, "is_genai": False},
    17: {"name": "Weather", "category": "utility", "memory_mb": 60, "is_genai": False},
    18: {"name": "AI_Assistant", "category": "genai", "memory_mb": 800, "is_genai": True},
    19: {"name": "GenAI_ImageGen", "category": "genai", "memory_mb": 1200, "is_genai": True},
}
NUM_APPS = len(APPS)

def _get_time_slot(hour):
    if 6 <= hour < 9: return "early_morning"
    elif 9 <= hour < 12: return "morning"
    elif 12 <= hour < 14: return "lunch"
    elif 14 <= hour < 18: return "afternoon"
    elif 18 <= hour < 21: return "evening"
    elif 21 <= hour < 24: return "night"
    else: return "late_night"

def _build_base_transition_matrix():
    T = np.ones((NUM_APPS, NUM_APPS)) * 0.1  # very low uniform baseline
    # Strong sequential patterns that real users exhibit
    pairs = [(4,5,25),(11,12,25),(2,15,15),(9,0,20),(3,4,20),(7,11,15),
             (6,7,15),(12,7,18),(12,9,15),(15,3,12),(5,0,15),(1,9,20),
             (9,1,15),(2,6,12),(18,2,18),(18,15,20),(2,18,12),
             (0,1,15),(1,0,10),(3,2,10),(7,8,12),(8,7,12),
             (6,10,10),(10,6,10),(16,17,20),(17,3,12)]
    for s,d,w in pairs:
        T[s,d] = w
    for i in range(NUM_APPS):
        T[i,i] = 0.05
    return T / T.sum(axis=1, keepdims=True)

def _get_time_bias(hour):
    bias = np.ones(NUM_APPS) * 0.3  # low baseline
    slot = _get_time_slot(hour)
    if slot == "early_morning":
        bias[[16,17,3,8,2]] = [20,15,10,5,5]
    elif slot == "morning":
        bias[[3,4,2,15,18]] = [18,15,12,10,15]
    elif slot == "lunch":
        bias[[6,7,9,10]] = [18,15,12,12]
    elif slot == "afternoon":
        bias[[3,2,18,15,5]] = [15,15,18,12,10]
    elif slot == "evening":
        bias[[6,7,8,10,9,19]] = [20,18,15,15,12,8]
    elif slot == "night":
        bias[[6,7,10,16]] = [18,15,12,15]
    else:
        bias[[6,16,10]] = [8,18,8]
    return bias / bias.sum()

def _get_weekday_modifier(dow):
    mod = np.ones(NUM_APPS) * 0.5
    if dow < 5:
        mod[[3,4,15,18,2]] = [5,5,4,6,3]
    else:
        mod[[6,7,11,12,5,19]] = [6,5,4,4,4,4]
    return mod / mod.sum()


class AppUsageDataGenerator:
    def __init__(self, num_users=50, seed=42):
        self.num_users = num_users
        self.rng = np.random.RandomState(seed)
        self.base_T = _build_base_transition_matrix()
        self.profiles = self._create_profiles()

    def _create_profiles(self):
        archetypes = ["power_user","social_butterfly","casual","content_consumer","professional"]
        profiles = []
        for uid in range(self.num_users):
            arch = archetypes[uid % len(archetypes)]
            noise = self.rng.dirichlet(np.ones(NUM_APPS)*20, size=NUM_APPS)
            pT = 0.9*self.base_T + 0.1*noise  # 90% base patterns, 10% noise
            pT = pT / pT.sum(axis=1, keepdims=True)
            pref = np.ones(NUM_APPS)
            if arch == "power_user": pref[[2,3,4,15,18,19]] = 3.0
            elif arch == "social_butterfly": pref[[7,8,9,11,12]] = 3.0
            elif arch == "casual": pref[[0,1,6,9,10]] = 2.0
            elif arch == "content_consumer": pref[[6,7,8,10,2]] = 3.0
            elif arch == "professional": pref[[3,4,5,15,18]] = 3.5
            pref /= pref.sum()
            profiles.append({"user_id":uid,"archetype":arch,"T":pT,"prefs":pref,
                           "daily_switches":self.rng.randint(40,120)})
        return profiles

    def generate(self, days=30):
        records = []
        for p in self.profiles:
            uid, T, prefs = p["user_id"], p["T"], p["prefs"]
            daily_n = p["daily_switches"]
            start = datetime(2025,1,1,8,0,0)
            cur = self.rng.choice(NUM_APPS, p=prefs)
            for day in range(days):
                dt = start + timedelta(days=day)
                dow = dt.weekday()
                n = max(20, daily_n + self.rng.randint(-10,10))
                hours = sorted(self.rng.choice(range(7,24), size=n, replace=True))
                for h in hours:
                    m, s = self.rng.randint(0,60), self.rng.randint(0,60)
                    ts = dt.replace(hour=h, minute=m, second=s)
                    tb = _get_time_bias(h)
                    wm = _get_weekday_modifier(dow)
                    probs = T[cur]*tb*wm*prefs
                    probs /= probs.sum()
                    nxt = self.rng.choice(NUM_APPS, p=probs)
                    info = APPS[nxt]
                    dur = (self.rng.exponential(120)+30 if info["is_genai"]
                           else self.rng.exponential(180)+60 if info["category"]=="entertainment"
                           else self.rng.exponential(60)+10 if info["category"]=="communication"
                           else self.rng.exponential(45)+5)
                    records.append({"user_id":uid,"timestamp":ts,"app_id":nxt,
                        "app_name":info["name"],"category":info["category"],
                        "memory_mb":info["memory_mb"],"hour":h,"minute":m,
                        "day_of_week":dow,"time_slot":_get_time_slot(h),
                        "is_genai":info["is_genai"],"session_duration_sec":round(dur,1)})
                    cur = nxt
        df = pd.DataFrame(records).sort_values(["user_id","timestamp"]).reset_index(drop=True)
        return df


def generate_kv_cache_workload(n=500, seed=42):
    rng = np.random.RandomState(seed)
    models = [
        {"name":"samsung_ai_small","base_mb":50,"max_ctx":2048},
        {"name":"samsung_ai_medium","base_mb":200,"max_ctx":4096},
        {"name":"samsung_ai_large","base_mb":500,"max_ctx":8192},
        {"name":"image_gen_diffusion","base_mb":800,"max_ctx":1024},
    ]
    records = []
    t0 = datetime(2025,1,1,8,0,0)
    for i in range(n):
        m = models[rng.choice(4, p=[.4,.3,.15,.15])]
        ctx = rng.randint(128, m["max_ctx"])
        kv = max(10, round(m["base_mb"]*(ctx/m["max_ctx"])+rng.normal(0,10),1))
        records.append({"request_id":i,"timestamp":t0+timedelta(seconds=i*rng.exponential(30)),
            "model_type":m["name"],"context_length":ctx,"kv_cache_size_mb":kv,
            "priority":rng.choice([1,2,3],p=[.2,.5,.3]),
            "is_continuation":rng.random()<0.4})
    return pd.DataFrame(records)


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    print("="*60+"\n  Generating Synthetic App Usage Dataset\n"+"="*60)
    gen = AppUsageDataGenerator(num_users=50, seed=42)
    df = gen.generate(days=30)
    df.to_csv(os.path.join(data_dir,"app_usage_logs.csv"), index=False)
    print(f"Records: {len(df):,} | Users: {df['user_id'].nunique()} | Apps: {df['app_id'].nunique()}")
    with open(os.path.join(data_dir,"app_metadata.json"),"w") as f:
        json.dump(APPS, f, indent=2, default=str)
    kv = generate_kv_cache_workload(500, 42)
    kv.to_csv(os.path.join(data_dir,"kv_cache_workload.csv"), index=False)
    print(f"KV workload: {len(kv)} requests")
    print("All datasets generated!")
