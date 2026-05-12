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
        self.sequences, self.contexts, self.targets, self.user_ids = [], [], [], []
        for uid in df["user_id"].unique():
            udf = df[df["user_id"] == uid].sort_values("timestamp").reset_index(drop=True)
            apps = udf["app_id"].values
            hours = udf["hour"].values
            dows = udf["day_of_week"].values
            for i in range(len(apps) - seq_len):
                h, d = hours[i + seq_len], dows[i + seq_len]
                self.sequences.append(apps[i:i + seq_len])
                self.contexts.append([
                    np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                    np.sin(2*np.pi*d/7), np.cos(2*np.pi*d/7),
                ])
                self.targets.append(apps[i + seq_len])
                self.user_ids.append(uid)
        self.sequences = np.array(self.sequences)
        self.contexts = np.array(self.contexts, dtype=np.float32)
        self.targets = np.array(self.targets)
        self.user_ids = np.array(self.user_ids)

    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        return (torch.LongTensor(self.sequences[idx]),
                torch.FloatTensor(self.contexts[idx]),
                torch.LongTensor([self.user_ids[idx]]),
                torch.LongTensor([self.targets[idx]]))


class NextAppLSTM(nn.Module):
    def __init__(self, num_apps=NUM_APPS, num_users=50, embed_dim=48,
                 user_embed_dim=16, hidden_dim=128, num_layers=2,
                 context_dim=CONTEXT_FEATURES, dropout=0.3):
        super().__init__()
        self.num_users = num_users
        self.app_embedding = nn.Embedding(num_apps, embed_dim)
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attn = nn.Linear(hidden_dim, 1)
        fc_input = hidden_dim + context_dim + user_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_input, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_apps),
        )

    def forward(self, seq, ctx, user_id=None):
        emb = self.app_embedding(seq)
        out, _ = self.lstm(emb)
        attn_w = torch.softmax(self.attn(out), dim=1)
        attended = (out * attn_w).sum(dim=1)
        parts = [attended, ctx]
        if user_id is not None:
            uid = user_id.squeeze(-1).clamp(0, self.num_users - 1)
            parts.append(self.user_embedding(uid))
        else:
            parts.append(torch.zeros(attended.size(0), 16, device=attended.device))
        return self.fc(torch.cat(parts, dim=1))

    def predict_proba(self, seq, ctx, user_id=None):
        with torch.no_grad():
            return torch.softmax(self.forward(seq, ctx, user_id), dim=1)

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

    def predict_proba(self, user_id, recent_apps, hour, dow):
        seq = torch.LongTensor([recent_apps[-SEQ_LEN:]]).to(self.device)
        ctx = torch.FloatTensor([[
            np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
            np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
        ]]).to(self.device)
        uid_t = torch.LongTensor([[user_id]]).to(self.device)
        lstm_p = self.lstm.predict_proba(seq, ctx, uid_t).cpu().numpy()[0]
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
        for seq, ctx, uid, target in train_loader:
            seq, ctx, uid = seq.to(device), ctx.to(device), uid.to(device)
            target = target.squeeze().to(device)
            optimizer.zero_grad()
            logits = model(seq, ctx, uid)
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
            for seq, ctx, uid, target in val_loader:
                seq, ctx, uid = seq.to(device), ctx.to(device), uid.to(device)
                target = target.squeeze().to(device)
                logits = model(seq, ctx, uid)
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
