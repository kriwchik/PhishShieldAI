# train_cnn_lstm.py
# Финальная версия: обучение CNN+BiLSTM модели на большом URL_dataset
# Поддержка Excel-кодировки, разделителя ',', обрезка мусорных столбцов
# Графики на русском, ускоренная тренировка

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report
)

from tqdm import tqdm

# ==========================
# ПУТИ
# ==========================
DATASET_PATH = "../dataset/URL_dataset.csv"
MODEL_DIR = "../model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PTH = os.path.join(MODEL_DIR, "url_cnn_lstm.pth")
TOKEN_MAP_PATH = os.path.join(MODEL_DIR, "token_map.json")
CLASS_MAP_PATH = os.path.join(MODEL_DIR, "class_map.json")

# ==========================
# ГИПЕРПАРАМЕТРЫ
# ==========================
MAX_LEN = 160
BATCH_SIZE = 256
EPOCHS = 8
LR = 1e-3
WEIGHT_DECAY = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# ==========================
# ЗАГРУЗКА ДАННЫХ
# ==========================
print("Загрузка датасета...")

raw = pd.read_csv(
    DATASET_PATH,
    sep=",",
    encoding="latin1",
    on_bad_lines="skip",
    header=0
)

# Чистим строки: берём только первые 2 значения
clean_rows = []
for row in raw.itertuples(index=False):
    url = str(row[0]).strip()
    label = str(row[1]).split(";")[0].strip()  # ОТРЕЗАЕМ мусор
    clean_rows.append((url, label))

df = pd.DataFrame(clean_rows, columns=["url", "label"])

df = df[df["label"].isin(["legitimate", "phishing"])]
df = df.dropna()
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

print("Всего строк:", len(df))
print(df["label"].value_counts())

# ==========================
# SPLIT
# ==========================
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
)

print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

# ==========================
# СЛОВАРЬ СИМВОЛОВ
# ==========================
all_text = "".join(train_df["url"].tolist())
chars = sorted(list(set(all_text)))

token_map = {"<PAD>": 0, "<UNK>": 1}
for i, ch in enumerate(chars, start=2):
    token_map[ch] = i

with open(TOKEN_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(token_map, f, ensure_ascii=False)

vocab_size = len(token_map)
print("Размер словаря:", vocab_size)

def encode_url(url: str):
    url = str(url)[:MAX_LEN]
    ids = [token_map.get(c, 1) for c in url]
    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    return np.array(ids[:MAX_LEN], dtype=np.int64)

# ==========================
# DATASET
# ==========================
class URLDataset(Dataset):
    def __init__(self, df):
        self.urls = df["url"].values
        self.labels = df["label"].map({"legitimate": 0, "phishing": 1}).values

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        x = encode_url(self.urls[idx])
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)

train_loader = DataLoader(URLDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(URLDataset(val_df), batch_size=BATCH_SIZE)
test_loader = DataLoader(URLDataset(test_df), batch_size=BATCH_SIZE)

# ==========================
# МОДЕЛЬ
# ==========================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        scores = self.attn(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context

class CNN_BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.embedding(x)
        c = torch.relu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        lstm_out, _ = self.lstm(c)
        context = self.attn(lstm_out)
        return self.fc(self.dropout(context))

model = CNN_BiLSTM(vocab_size).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ==========================
# ТРЕНИРОВКА
# ==========================
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(train_loader, desc=f"Эпоха {epoch}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    train_losses.append(total_loss / len(train_loader))
    train_accs.append(correct / total)

    # Валидация
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()
            val_correct += (out.argmax(1) == y).sum().item()
            val_total += y.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(val_correct / val_total)

    print(f"Эпоха {epoch}: Train loss={train_losses[-1]:.4f}, Val loss={val_losses[-1]:.4f}")

torch.save(model.state_dict(), MODEL_PTH)

# ==========================
# ГРАФИКИ
# ==========================
sns.set(style="whitegrid", font_scale=1.2)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Потери (train)")
plt.plot(val_losses, label="Потери (val)")
plt.title("График функции потерь")
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "loss.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(train_accs, label="Точность (train)")
plt.plot(val_accs, label="Точность (val)")
plt.title("График точности")
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "accuracy.png"))
plt.close()

print("Обучение завершено. Модель и графики сохранены в папке model/")
