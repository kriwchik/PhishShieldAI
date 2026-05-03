from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import json
import numpy as np
from urllib.parse import urlparse

MAX_LEN = 160
MODEL_DIR = "../model"
MODEL_PTH = f"{MODEL_DIR}/url_cnn_lstm.pth"
TOKEN_MAP_PATH = f"{MODEL_DIR}/token_map.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Нормализация URL
# ---------------------------
def normalize_url(url: str) -> str:
    url = url.strip()

    if url.startswith("url="):
        url = url[4:]

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    parsed = urlparse(url)
    host = parsed.netloc

    if not host.startswith("www."):
        host = "www." + host

    return f"{parsed.scheme}://{host}{parsed.path}"


# ---------------------------
# Модель
# ---------------------------
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


# ---------------------------
# Загрузка токенов и модели
# ---------------------------
with open(TOKEN_MAP_PATH, "r", encoding="utf-8") as f:
    token_map = json.load(f)

vocab_size = len(token_map)

def encode_url(url: str):
    url = str(url)[:MAX_LEN]
    ids = [token_map.get(c, 1) for c in url]
    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    return np.array(ids[:MAX_LEN], dtype=np.int64)

model = CNN_BiLSTM(vocab_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PTH, map_location=DEVICE))
model.eval()


# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI()

class URLRequest(BaseModel):
    url: str

@app.post("/analyze")
def analyze(data: URLRequest):
    normalized = normalize_url(data.url)
    x = torch.tensor(encode_url(normalized)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

    return {
        "normalized_url": normalized,
        "legitimate": float(probs[0] * 100),
        "phishing": float(probs[1] * 100)
    }
