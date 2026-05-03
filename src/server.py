import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Модель CNN + BiLSTM + Attention
# -----------------------------
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # embedding 161 × 128
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)

        # conv 128 → 128 kernel=5
        self.conv = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=5,
            padding=2
        )

        # BiLSTM hidden=128
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # attention: 256 → 1
        self.attn = nn.Linear(256, 1)

        # final classifier: 256 → 2
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.embedding(x)          # [B, L, 128]
        x = x.transpose(1, 2)          # [B, 128, L]
        x = F.relu(self.conv(x))       # [B, 128, L]
        x = x.transpose(1, 2)          # [B, L, 128]

        lstm_out, _ = self.lstm(x)     # [B, L, 256]

        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # [B, L, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)       # [B, 256]

        return self.fc(context)

# -----------------------------
# Пути
# -----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "url_cnn_lstm.pth")
TOKEN_MAP_PATH = os.path.join(MODEL_DIR, "token_map.json")

# -----------------------------
# Токены
# -----------------------------
with open(TOKEN_MAP_PATH, "r", encoding="utf-8") as f:
    token_map = json.load(f)

vocab_size = len(token_map)

# -----------------------------
# Загрузка модели
# -----------------------------
device = torch.device("cpu")

model = CNN_LSTM_Attention(vocab_size)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

# -----------------------------
# Нормализация URL
# -----------------------------
def normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"https?://", url):
        url = "https://" + url
    return url

# -----------------------------
# Токенизация
# -----------------------------
def tokenize(url: str):
    tokens = [token_map.get(ch, 1) for ch in url]
    if len(tokens) > 200:
        tokens = tokens[:200]
    else:
        tokens += [0] * (200 - len(tokens))
    return torch.tensor([tokens], dtype=torch.long)

# -----------------------------
# Pydantic
# -----------------------------
class URLRequest(BaseModel):
    url: str

# -----------------------------
# API
# -----------------------------
@app.post("/analyze")
def analyze(req: URLRequest):
    url = normalize_url(req.url)
    x = tokenize(url).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    return {
        "normalized_url": url,
        "legitimate": float(probs[0] * 100),
        "phishing": float(probs[1] * 100)
    }

@app.get("/")
def root():
    return {"status": "PhishShieldAI backend running"}
