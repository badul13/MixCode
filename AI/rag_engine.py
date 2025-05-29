import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 뉴스 데이터 로딩
news_data = pd.read_csv('/content/opendata_output.csv', usecols=['content'])
news_data = news_data.dropna(subset=['content'])
documents = news_data['content'].tolist()

# 벡터화 (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(documents)

# 생성 모델 로딩 (KoBART)
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


def vector_store(query, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [documents[i] for i in top_indices]
    return " ".join(results)


def generate_answer(context, question):
    input_text = f"{context} 질문: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
    summary_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
