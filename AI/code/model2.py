import os
import re
import time
import torch
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
from urllib.parse import quote
from dotenv import load_dotenv
from typing import List
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# ✅ 환경 변수 로드
load_dotenv()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
assert NAVER_CLIENT_ID, "❗ .env 파일에서 NAVER_CLIENT_ID를 불러올 수 없습니다."
assert NAVER_CLIENT_SECRET, "❗ .env 파일에서 NAVER_CLIENT_SECRET를 불러올 수 없습니다."

# ✅ fetch 유틸
def fetch_with_retry(url, headers=None, max_retries=3):
    for _ in range(max_retries):
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                return res
        except requests.RequestException:
            time.sleep(1)
    return None

# ✅ Dataset 클래스
class SummaryDataset(Dataset):
    def __init__(self, summaries, labels, tokenizer, max_len=256):
        self.summaries = summaries    
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.summaries)    

    def __getitem__(self, idx):
        text = self.summaries[idx]    
        label = self.labels[idx]
        encoding = self.tokenizer(
          text,
          truncation=True,
          padding='max_length',
          max_length=self.max_len,
          return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ✅ 모델 클래스
class KoBERTClassifier(torch.nn.Module):
    def __init__(self, bert):
        super(KoBERTClassifier, self).__init__()
        self.bert = bert
        self.classifier = torch.nn.Linear(bert.config.hidden_size, 2)  # 이진 분류

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled_output)

# ✅ 학습 유틸
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for i, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 10 == 0 or (i + 1) == len(loader):
            percent = ((i + 1) / len(loader)) * 100
            print(f"🌀 [{i + 1}/{len(loader)}] ({percent:.1f}%) Loss: {loss.item():.4f}")
    return total_loss / len(loader)

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def plot_training_history(train_losses, val_accuracies):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

# ✅ 데이터 로드
def load_data_from_csv(path):
    df = pd.read_csv(path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return list(zip(texts, labels))

# ✅ 전처리
def prepare_rag_training_data(raw_data, force=False):
    if not raw_data:
        return [], []
    texts, labels = zip(*raw_data)
    return list(texts), list(labels)

    print("🧪 모델 학습 시작...")
    print(f"총 학습 데이터 수: {len(train_dataset)}")
    print(f"배치 크기: {train_loader.batch_size}, 총 배치 수: {len(train_loader)}")

# ✅ 메인 실행
def main(force=False):
    print("📂 데이터 로드 중...")
    raw_data = load_data_from_csv("outputdata.csv")
    texts, labels = prepare_rag_training_data(raw_data, force=force)
    if not texts:
        print("❗ 학습할 데이터가 부족합니다.")
        return

    print("📊 학습/검증 데이터 분할...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print("🔤 토크나이저 및 모델 불러오기...")
    tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')
    bert_model = BertModel.from_pretrained('beomi/kcbert-base')

    train_dataset = SummaryDataset(train_texts, train_labels, tokenizer)
    val_dataset = SummaryDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KoBERTClassifier(bert_model).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, val_accuracies = [], []
    epochs = 3
    for epoch in range(epochs):
        print(f"\n📘 Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        print(f"✅ Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | 🔍 Validation Acc: {val_acc:.4f}")
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        torch.save(model.state_dict(), f"rag_classifier_model.pt")
        print("💾 모델 저장 완료: rag_classifier_model.pt")

    plot_training_history(train_losses, val_accuracies)

if __name__ == "__main__":
    main()
