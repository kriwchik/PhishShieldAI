import os
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import re

# -----------------------------
# 1. FastAPI + CORS
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Разрешаем запросы с любых сайтов
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 2. Модель
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# -----------------------------
# 3. Пути к файлам модели
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "url_cnn_lstm.pth")  # <-- правильное имя файла
TOKEN_MAP_PATH = os.path.join(MODEL_DIR, "token_map.json")

# -----------------------------
# 4. Загрузка токенов
# -----------------------------
with open(TOKEN_MAP_PATH, "r", encoding="utf-8") as f:
    token_map = json.load(f)

vocab_size = len(token_map)

# -----------------------------
# 5. Загрузка модели
# -----------------------------
device = torch.device("cpu")

model = LSTMClassifier(vocab_size=vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# 6. Нормализация URL
# -----------------------------
def normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"https?://", url):
        url = "https://" + url
    return url

# -----------------------------
# 7. Токенизация
# -----------------------------
def tokenize(url: str):
    tokens = [token_map.get(ch, 1) for ch in url]  # 1 = unknown
    if len(tokens) > 200:
        tokens = tokens[:200]
    else:
        tokens += [0] * (200 - len(tokens))
    return torch.tensor([tokens], dtype=torch.long)

# -----------------------------
# 8. Pydantic модель
# -----------------------------
class URLRequest(BaseModel):
    url: str

# -----------------------------
# 9. Маршрут анализа
# -----------------------------
@app.post("/analyze")
def analyze(request: URLRequest):
    url = normalize_url(request.url)
    x = tokenize(url).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    legitimate = float(probs[0] * 100)
    phishing = float(probs[1] * 100)

    return {
        "normalized_url": url,
        "legitimate": legitimate,
        "phishing": phishing
    }

# -----------------------------
# 10. Корневой маршрут
# -----------------------------
@app.get("/")
def root():
    return {"status": "PhishGuar backend is running"}
