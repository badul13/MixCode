import os
import re
import time
import requests
import pandas as pd
from typing import List
from urllib.parse import quote
from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from dotenv import load_dotenv
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from torch.cuda.amp import autocast, GradScaler

class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index]['text'])
        label = int(self.data.iloc[index]['label'])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def prepare_datasets(csv_path):
    df = pd.read_csv(csv_path)

    if df['label'].dtype == object:
        df['label'] = df['label'].map({'fake': 0, 'real': 1})

    df.dropna(subset=['text', 'label'], inplace=True)
    return df

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    scaler = GradScaler()

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

def plot_training_history(train_losses, val_accuracies):
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Train loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o')
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Validation accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, marker='o', color='green')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()

def main():
    csv_path = '/content/outputdata.csv'
    full_df = prepare_datasets(csv_path)

    train_df, val_df = train_test_split(full_df, test_size=0.2, stratify=full_df['label'], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    config = AutoConfig.from_pretrained("beomi/KcELECTRA-base", num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = NewsDataset(train_df, tokenizer)
    val_dataset = NewsDataset(val_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    train_losses = []
    val_accuracies = []

    epochs = 3
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_accuracy = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

    torch.save(model.state_dict(), "/content/news_classifier_model.pt")
    print(f"💾 모델 저장 완료: /content/news_classifier_model.pt")
    print("\n🎉 훈련 완료, 결과 시각화 중...")
    plot_training_history(train_losses, val_accuracies)

if __name__ == "__main__":
    main()
