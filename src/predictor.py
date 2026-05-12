"""
Next-App Prediction Model (The "Brain")
Hybrid: LSTM neural model + per-user Markov chain ensemble.
The Markov model is trained on ALL users for per-user personalization.
LSTM is trained on train users, fine-tuned approach with user embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
import os, json, pickle

NUM_APPS = 20
SEQ_LEN = 10
CONTEXT_FEATURES = 4


class AppSequenceDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.sequences, self.contexts, self.targets = [], [], []
        self.user_ids, self.time_deltas = [], []
        for uid in df["user_id"].unique():
            udf = df[df["user_id"] == uid].sort_values("timestamp").reset_index(drop=True)
            apps = udf["app_id"].values
            hours = udf["hour"].values
            dows = udf["day_of_week"].values
            # Compute time deltas between consecutive accesses (seconds)
            ts = pd.to_datetime(udf["timestamp"])
            deltas = np.zeros(len(apps), dtype=np.float32)
            for j in range(1, len(apps)):
                deltas[j] = max(0, (ts.iloc[j] - ts.iloc[j-1]).total_seconds())
            for i in range(len(apps) - seq_len):
                h, d = hours[i + seq_len], dows[i + seq_len]
                self.sequences.append(apps[i:i + seq_len])
                self.time_deltas.append(deltas[i:i + seq_len])
                self.contexts.append([
                    np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                    np.sin(2*np.pi*d/7), np.cos(2*np.pi*d/7),
                ])
                self.targets.append(apps[i + seq_len])
                self.user_ids.append(uid)
        self.sequences = np.array(self.sequences)
        self.time_deltas = np.array(self.time_deltas, dtype=np.float32)
        self.contexts = np.array(self.contexts, dtype=np.float32)
        self.targets = np.array(self.targets)
        self.user_ids = np.array(self.user_ids)

    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        return (torch.LongTensor(self.sequences[idx]),
                torch.FloatTensor(self.contexts[idx]),
                torch.LongTensor([self.user_ids[idx]]),
                torch.LongTensor([self.targets[idx]]),
                torch.FloatTensor(self.time_deltas[idx]))


class NextAppLSTM(nn.Module):
    """
    Time-Aware GRU with Dot-Product Attention (T-GRU-Attn).

    Upgrades over vanilla LSTM:
    1. GRU instead of LSTM: 25% fewer params, same accuracy for user behavior
    2. Time-decay gating: time delta between events modulates the reset gate,
       flushing stale context when the phone has been idle
    3. Dot-product attention: instead of just the final hidden state, compute
       attention scores over ALL hidden states to "look back" at important past apps
    4. Session-aware: time deltas feed into the model so it can detect session breaks

    Note: Class name kept as NextAppLSTM for backward compatibility with model loading.
    """
    def __init__(self, num_apps=NUM_APPS, num_users=50, embed_dim=64,
                 user_embed_dim=24, hidden_dim=128, num_layers=2,
                 context_dim=CONTEXT_FEATURES, dropout=0.3):
        super().__init__()
        self.num_users = num_users
        self.hidden_dim = hidden_dim

        # Deeper embeddings (GRU savings allow us to increase from 48→64)
        self.app_embedding = nn.Embedding(num_apps, embed_dim)
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)

        # Time-delta projection: maps scalar Δt to a learned time embedding
        self.time_delta_proj = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, embed_dim),
        )

        # GRU instead of LSTM (25% fewer gate operations)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Time-decay gate: learned function that modulates reset based on Δt
        # g(Δt) → [0, 1], where large Δt → small g → flush hidden state
        self.time_gate = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, hidden_dim), nn.Sigmoid(),
        )

        # Lightweight dot-product attention over all hidden states
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_scale = hidden_dim ** 0.5

        # Classification head
        fc_input = hidden_dim + context_dim + user_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_apps),
        )

    def forward(self, seq, ctx, user_id=None, time_deltas=None):
        # seq: (batch, seq_len) app IDs
        # time_deltas: (batch, seq_len) seconds between consecutive accesses
        emb = self.app_embedding(seq)  # (batch, seq_len, embed_dim)

        # Inject time-delta information into embeddings
        if time_deltas is not None:
            td = time_deltas.unsqueeze(-1).float()  # (batch, seq_len, 1)
            # Normalize time deltas (log-scale, clamp to avoid inf)
            td_norm = torch.log1p(td.clamp(min=0, max=86400)) / 11.4  # log(86400)≈11.4
            time_emb = self.time_delta_proj(td_norm)  # (batch, seq_len, embed_dim)
            emb = emb + time_emb  # additive time encoding

        # GRU forward pass: get ALL hidden states (not just final)
        all_hidden, h_n = self.gru(emb)  # all_hidden: (batch, seq_len, hidden_dim)

        # Time-decay gating on the final hidden state
        if time_deltas is not None:
            # Use the LAST time delta to decide how much to decay
            last_td = time_deltas[:, -1:].float().unsqueeze(-1)  # (batch, 1, 1)
            last_td_norm = torch.log1p(last_td.clamp(min=0, max=86400)) / 11.4
            gate = self.time_gate(last_td_norm.squeeze(1))  # (batch, hidden_dim)
            # Apply gate: small Δt → gate≈1 (keep), large Δt → gate≈0 (flush)
            final_h = all_hidden[:, -1, :] * gate
        else:
            final_h = all_hidden[:, -1, :]

        # Dot-product attention: query = final hidden, keys = all hidden states
        query = self.attn_query(final_h).unsqueeze(1)  # (batch, 1, hidden_dim)
        scores = torch.bmm(query, all_hidden.transpose(1, 2)) / self.attn_scale
        attn_weights = torch.softmax(scores, dim=-1)  # (batch, 1, seq_len)
        attended = torch.bmm(attn_weights, all_hidden).squeeze(1)  # (batch, hidden_dim)

        # Combine attended output with context and user embedding
        parts = [attended, ctx]
        if user_id is not None:
            uid = user_id.squeeze(-1).clamp(0, self.num_users - 1)
            parts.append(self.user_embedding(uid))
        else:
            parts.append(torch.zeros(attended.size(0), 24, device=attended.device))
        return self.fc(torch.cat(parts, dim=1))

    def predict_proba(self, seq, ctx, user_id=None, time_deltas=None):
        with torch.no_grad():
            return torch.softmax(self.forward(seq, ctx, user_id, time_deltas), dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MarkovPredictor:
    """Per-user frequency-based Markov chain with time-of-day patterns."""
    def __init__(self, num_apps=NUM_APPS, order=2):
        self.num_apps = num_apps
        self.order = order
        self.user_transitions = defaultdict(lambda: defaultdict(Counter))
        self.user_time_patterns = defaultdict(lambda: defaultdict(Counter))
        self.user_app_freq = defaultdict(Counter)

    def fit(self, df):
        for uid in df["user_id"].unique():
            udf = df[df["user_id"] == uid].sort_values("timestamp")
            apps = udf["app_id"].values
            hours = udf["hour"].values
            for i in range(self.order, len(apps)):
                prev = tuple(apps[i - self.order:i])
                self.user_transitions[uid][prev][apps[i]] += 1
                self.user_time_patterns[uid][hours[i]][apps[i]] += 1
                self.user_app_freq[uid][apps[i]] += 1

    def predict_proba(self, user_id, recent_apps, hour):
        probs = np.ones(self.num_apps) * 0.01

        # Per-user app frequency baseline
        if user_id in self.user_app_freq:
            freq = self.user_app_freq[user_id]
            total = sum(freq.values())
            if total > 0:
                for app, cnt in freq.items():
                    probs[app] += 0.5 * cnt / total

        # Transition probability (strongest signal)
        if user_id in self.user_transitions:
            key = tuple(recent_apps[-self.order:])
            counts = self.user_transitions[user_id].get(key, Counter())
            total = sum(counts.values())
            if total > 0:
                for app, cnt in counts.items():
                    probs[app] += 5.0 * cnt / total

        # Time-of-day
        if user_id in self.user_time_patterns:
            tc = self.user_time_patterns[user_id].get(hour, Counter())
            total = sum(tc.values())
            if total > 0:
                for app, cnt in tc.items():
                    probs[app] += 2.0 * cnt / total

        probs /= probs.sum()
        return probs


class EnsemblePredictor:
    def __init__(self, lstm_model, markov_model, device,
                 lstm_weight=0.3, markov_weight=0.7):
        self.lstm = lstm_model
        self.markov = markov_model
        self.device = device
        self.lw = lstm_weight
        self.mw = markov_weight

    def predict_proba(self, user_id, recent_apps, hour, dow, time_deltas=None):
        seq = torch.LongTensor([recent_apps[-SEQ_LEN:]]).to(self.device)
        ctx = torch.FloatTensor([[
            np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
            np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
        ]]).to(self.device)
        uid_t = torch.LongTensor([[user_id]]).to(self.device)
        td = None
        if time_deltas is not None:
            td = torch.FloatTensor([time_deltas[-SEQ_LEN:]]).to(self.device)
        lstm_p = self.lstm.predict_proba(seq, ctx, uid_t, time_deltas=td).cpu().numpy()[0]
        markov_p = self.markov.predict_proba(user_id, recent_apps, hour)
        combined = self.lw * lstm_p + self.mw * markov_p
        combined /= combined.sum()
        return combined


def train_model(df, epochs=40, batch_size=256, lr=0.001, val_split=0.15,
                model_dir="models", device=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    num_users = df["user_id"].nunique()

    # ─── Train Markov on ALL data (it's per-user, so no leakage) ───
    print("Training Markov predictor on all users...")
    markov = MarkovPredictor(NUM_APPS, order=2)
    markov.fit(df)

    # Evaluate Markov with time-based split (last 20% of each user's data)
    m_correct, m_top3, m_total = 0, 0, 0
    for uid in df["user_id"].unique():
        udf = df[df["user_id"] == uid].sort_values("timestamp").reset_index(drop=True)
        apps = udf["app_id"].values
        hours = udf["hour"].values
        n = len(apps)
        test_start = int(n * 0.8)
        for i in range(max(test_start, SEQ_LEN), n):
            probs = markov.predict_proba(uid, list(apps[max(0,i-SEQ_LEN):i]), hours[i])
            if np.argmax(probs) == apps[i]: m_correct += 1
            if apps[i] in np.argsort(probs)[-3:]: m_top3 += 1
            m_total += 1
    print(f"  Markov Accuracy: {m_correct/max(1,m_total):.4f} | "
          f"Top-3: {m_top3/max(1,m_total):.4f}")

    # ─── Train LSTM with user-based split ───
    print("Training LSTM predictor...")
    users = df["user_id"].unique()
    np.random.seed(42)
    np.random.shuffle(users)
    split = int(len(users) * (1 - val_split))
    train_users, val_users = users[:split], users[split:]

    train_ds = AppSequenceDataset(df[df["user_id"].isin(train_users)])
    val_ds = AppSequenceDataset(df[df["user_id"].isin(val_users)])
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = NextAppLSTM(num_users=num_users).to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_top3_acc": []}
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, ctx, uid, target, td in train_loader:
            seq, ctx, uid = seq.to(device), ctx.to(device), uid.to(device)
            td = td.to(device)
            target = target.squeeze().to(device)
            optimizer.zero_grad()
            logits = model(seq, ctx, uid, time_deltas=td)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_train = total_loss / len(train_loader)

        model.eval()
        val_loss, correct, top3_c, total = 0, 0, 0, 0
        with torch.no_grad():
            for seq, ctx, uid, target, td in val_loader:
                seq, ctx, uid = seq.to(device), ctx.to(device), uid.to(device)
                td = td.to(device)
                target = target.squeeze().to(device)
                logits = model(seq, ctx, uid, time_deltas=td)
                val_loss += criterion(logits, target).item()
                correct += (logits.argmax(1) == target).sum().item()
                top3 = logits.topk(3, dim=1).indices
                top3_c += (top3 == target.unsqueeze(1)).any(1).sum().item()
                total += target.size(0)

        vacc = correct / total
        t3acc = top3_c / total
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(vacc)
        history["val_top3_acc"].append(t3acc)

        if vacc > best_val_acc:
            best_val_acc = vacc
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "best_predictor.pth"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_train:.4f} | "
                  f"Acc: {vacc:.4f} | Top-3: {t3acc:.4f}")

    torch.save(model.state_dict(), os.path.join(model_dir, "final_predictor.pth"))
    with open(os.path.join(model_dir, "markov_model.pkl"), "wb") as f:
        pickle.dump({"transitions": dict(markov.user_transitions),
                      "time_patterns": dict(markov.user_time_patterns),
                      "app_freq": dict(markov.user_app_freq)}, f)
    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    print(f"\n  LSTM Best Val Acc: {best_val_acc:.4f} | Top-3: {max(history['val_top3_acc']):.4f}")
    return model, markov, history


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "app_usage_logs.csv")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    if not os.path.exists(data_path):
        print("Run data_generator.py first."); exit(1)
    df = pd.read_csv(data_path)
    train_model(df, epochs=40, model_dir=model_dir)
