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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

# 환경 변수 로드
load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    raise ValueError("네이버 API 키가 환경변수에서 로드되지 않았습니다. .env 파일을 확인하세요.")

# 재시도 함수
def fetch_with_retry(url, headers, retries=3, timeout=15, delay=2):
    for i in range(retries):
        try:
            res = requests.get(url, headers=headers, timeout=timeout)
            res.raise_for_status()
            return res
        except requests.exceptions.RequestException as e:
            print(f"🔴 요청 실패 {url} (시도 {i+1}/{retries}): {e}")
            if i < retries - 1:
                time.sleep(delay)
            else:
                return None

# 네이버 뉴스 크롤러
class NaverNewsCrawler:
    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }

    def extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[가-힣]{2,}\b', text)
        if not words:
            return ["뉴스"]
        counted = Counter(words)
        return [w for w, c in counted.most_common(5)]

    def search_news_urls(self, keywords: List[str], max_total=20, display=5) -> List[str]:
        if not keywords:
            print("키워드가 없습니다.")
            return []

        query = quote(" ".join(keywords))
        all_links = set()
        start = 1

        while len(all_links) < max_total and start <= 100:
            url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display={display}&start={start}&sort=sim"
            try:
                res = requests.get(url, headers=self.headers, timeout=10)
                if res.status_code != 200:
                    print(f"🔴 네이버 API 요청 실패: {res.status_code}")
                    break
                items = res.json().get("items", [])
                if not items:
                    break

                for item in items:
                    link = item.get("link", "")
                    if any(domain in link for domain in ["news.naver.com", "news.sbs.co.kr"]):
                        all_links.add(link)

                if len(items) < display:
                    break

                start += display
                time.sleep(0.3)
            except Exception as e:
                print(f"🔴 뉴스 검색 실패: {e}")
                break

        return list(all_links)[:max_total]

    def crawl_articles(self, urls: List[str]) -> List[str]:
        articles = []
        for url in urls:
            res = fetch_with_retry(url, headers={"User-Agent": "Mozilla/5.0"})
            if res is None:
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            selectors = [
                "div#newsct_article", "div#articleBodyContents", "div#articeBody",
                "div#newsEndContents", "div#dic_area", "div.article_body"
            ]
            article_text = ""
            for sel in selectors:
                container = soup.select_one(sel)
                if container:
                    article_text = container.get_text(separator=" ", strip=True)
                    break
            if len(article_text) > 100:
                articles.append(article_text)
        return articles

# 요약기
class RagSummarizer:
    def __init__(self, model_name="digit82/kobart-summarization"):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

    def summarize(self, texts: List[str], max_length=128) -> List[str]:
        summaries = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for text in texts:
            inputs = self.tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True).to(device)
            summary_ids = self.model.generate(
                inputs, max_length=max_length, min_length=30,
                length_penalty=2.0, num_beams=4, early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        return summaries

# KoBERT 분류기
class KoBERTClassifier(torch.nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=2):
        super(KoBERTClassifier, self).__init__()
        self.bert = bert
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

# Dataset
class SummaryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 학습 루프
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

# 요약 데이터 생성 or 캐시 로딩 (중복 제거 포함)
def prepare_rag_training_data(text_label_pairs, cache_path="rag_summaries.csv", force=False):
    if os.path.exists(cache_path) and not force:
        print("✅ 요약 캐시 파일 로딩 중...")
        df = pd.read_csv(cache_path)
        df = df.drop_duplicates(subset=["summary", "label"])
        return df["summary"].tolist(), df["label"].tolist()

    crawler = NaverNewsCrawler()
    summarizer = RagSummarizer()
    all_summaries, all_labels = [], []

    for raw_text, label in text_label_pairs:
        keywords = crawler.extract_keywords(raw_text)
        urls = crawler.search_news_urls(keywords, max_total=20)
        articles = crawler.crawl_articles(urls)
        if not articles:
            print(f"❗ 뉴스 없음: {raw_text}")
            continue
        summaries = summarizer.summarize(articles)
        all_summaries.extend(summaries)
        all_labels.extend([label] * len(summaries))

    df = pd.DataFrame({"summary": all_summaries, "label": all_labels})
    df.drop_duplicates(subset=["summary", "label"], inplace=True)
    df.to_csv(cache_path, index=False)
    print("✅ 요약 데이터 캐시에 저장 완료")
    return df["summary"].tolist(), df["label"].tolist()

def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    return list(zip(df['text'].tolist(), df['label'].tolist()))

# 체크포인트 저장
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"💾 체크포인트 저장: {path}")

# 체크포인트 불러오기 (학습 재개용)
def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"🔄 체크포인트 로딩 완료: epoch {epoch}, loss {loss:.4f}")
    return model, optimizer, epoch, loss

# 메인 함수
def main(force=False, resume_checkpoint=None):
    raw_data = load_data_from_csv("outputdata.csv")
    print(f"원본 데이터 개수: {len(raw_data)}")
    print(f"중복 제거 후 개수: {len(set(raw_data))}")
    print("[1] 뉴스 수집 및 요약")
    texts, labels = prepare_rag_training_data(raw_data, force=force)
    if not texts:
        print("❗ 학습할 데이터가 부족합니다.")
        return

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    bert_model = BertModel.from_pretrained('monologg/kobert')

    train_dataset = SummaryDataset(train_texts, train_labels, tokenizer)
    val_dataset = SummaryDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KoBERTClassifier(bert_model).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    train_losses, val_accuracies = [], []
    best_val_acc = 0.0

    # 체크포인트 불러오기 (재개)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, resume_checkpoint, device)
        print(f"🎉 학습 재개: {start_epoch+1} 에폭부터 시작")

    print("[2] 모델 학습 시작")
    for epoch in range(start_epoch, 3):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # 에폭별 체크포인트 저장
        save_checkpoint(model, optimizer, epoch, train_loss, f"rag_classifier_epoch{epoch+1}.pth")

        # 모델도 별도로 저장
        torch.save(model.state_dict(), "/content/rag_classifier_model.pt")

    print("\n✅ 학습 완료! 모델이 저장되었습니다.")
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_accuracies, label='val_acc')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    main(force=False, resume_checkpoint=None)
