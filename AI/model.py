import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.optim import AdamW
from tqdm import tqdm

# 1. CSV 로드 함수
def prepare_datasets(csv_path):
    df = pd.read_csv(csv_path)
    print("🔍 로드된 컬럼:", df.columns)
    print("🧾 상위 데이터:\n", df.head())

    if 'content' not in df.columns or 'label' not in df.columns:
        raise ValueError("❌ 'content' 또는 'label' 컬럼이 CSV에 없습니다.")

    df = df[['content', 'label']].dropna(subset=['content'])
    df = df[df['content'].str.strip() != '']

    if df.empty:
        raise ValueError("❌ content가 모두 비어 있습니다.")

    return df

# 2. Dataset 클래스
class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['content']
        label = self.data.iloc[idx]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 3. 훈련 함수
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", ncols=100):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 4. 검증 함수
def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = total_correct / total
    return accuracy

# 5. 메인 실행부
def main():
    csv_path = '/content/opendata_output.csv'
    full_df = prepare_datasets(csv_path)

    train_df, val_df = train_test_split(full_df, test_size=0.2, stratify=full_df['label'], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    config = AutoConfig.from_pretrained("beomi/KcELECTRA-base", num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base", config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_dataset = NewsDataset(train_df, tokenizer, max_len=128)
    val_dataset = NewsDataset(val_df, tokenizer, max_len=128)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 3
    for epoch in range(epochs):
        print(f"\n📘 Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"✅ Training Loss: {train_loss:.4f} | 🔍 Validation Acc: {val_acc:.4f}")
        torch.save(model.state_dict(), f"/content/news_classifier.pt")
        print(f"💾 모델 저장 완료: /content/news_classifier.pt")

    print("🎉 훈련 완료")

if __name__ == "__main__":
    main()
