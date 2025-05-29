import os
import re
import torch
import threading
import requests
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from typing import List
from dotenv import load_dotenv
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from naver_news_crawler import NaverNewsCrawler
from kobert_classifier import predict_prob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["NAVER_CLIENT_ID"] = "9wkHoQIN4RQakh958RQv"
os.environ["NAVER_CLIENT_SECRET"] = "xs9W_ozJ1g"

# Naver 뉴스 크롤러 초기화
crawler = NaverNewsCrawler()

# 환경 변수 로드
load_dotenv()

# Flask 앱 초기화
app = Flask(__name__)
run_with_ngrok(app)  # ngrok과 연결

# KoBART 요약 모델 초기화
tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 벡터검색 클래스
class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = None
        self.documents = []

    def build_index(self, docs: List[str]):
        self.documents = docs
        if docs:
            self.doc_vectors = self.vectorizer.fit_transform(docs)
        else:
            self.doc_vectors = None

    def search(self, query: str, top_k=3) -> List[str]:
        if self.doc_vectors is None or not self.documents:
            return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        if not similarities.any():
            return []
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = [self.documents[i] for i in top_indices if similarities[i] > 0]
        return results if results else []

vector_store = VectorStore()

# KoBART 요약 함수
def summarize_docs_korean(docs: List[str], max_docs: int = 3) -> List[str]:
    summaries = []
    for doc in docs[:max_docs]:
        try:
            input_ids = tokenizer.encode(doc[:1024], return_tensors="pt", max_length=1024, truncation=True).to(device)
            summary_ids = model.generate(
                input_ids,
                max_length=128,
                min_length=32,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"요약 실패:{e}")
    return summaries

# KoBART 기반 질문+문맥 답변 생성 함수
def generate_answer(question: str, contexts: List[str]) -> str:
    context_text = " ".join(contexts)
    input_text = f"{context_text} 질문: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
    summary_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Flask API: 뉴스 진위 예측 + RAG 응답
@app.route("/verify", methods=["POST"])
def verify_news():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "텍스트가 비어 있습니다."}), 400

        probs = predict_prob(text)
        real_prob, fake_prob = probs[0], probs[1]

        keywords = crawler.extract_keywords(text)
        urls = crawler.search_news_urls(keywords)
        docs = crawler.crawl_articles(urls)

        if not docs:
            return jsonify({
                "error": None,
                "real": round(real_prob, 3),
                "fake": round(fake_prob, 3),
                "keywords": keywords,
                "urls": urls,
                "ragAnswer": "",
                "confidence": 0.0,
                "summaries": []
            }), 200

        vector_store.build_index(docs)
        top_docs = vector_store.search(text, top_k=3)

        if not top_docs:
            return jsonify({
                "error": None,
                "real": round(real_prob, 3),
                "fake": round(fake_prob, 3),
                "keywords": keywords,
                "urls": urls,
                "ragAnswer": "⚠️ 관련 뉴스가 없습니다. 다른 문장으로 시도해주세요.",
                "confidence": 0.0,
                "summaries": []
            }), 200

        answer = generate_answer(text, top_docs)
        confidence = 0.8
        summaries = summarize_docs_korean(top_docs)

        return jsonify({
            "error": None,
            "real": round(real_prob, 3),
            "fake": round(fake_prob, 3),
            "keywords": keywords,
            "urls": urls,
            "ragAnswer": answer,
            "confidence": round(confidence, 3),
            "summaries": summaries
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def cli_loop():
    print("뉴스 진위 예측 + RAG 기반 뉴스 검색 챗봇 시작. 종료하려면 'exit' 입력\n")
    while True:
        text = input("문장을 입력해주세요: ").strip()
        if text.lower() == "exit":
            print("챗봇을 종료합니다.")
            break
        if not text:
            print("⚠️ 빈 입력입니다. 다시 시도하세요.\n")
            continue

        try:
            probs = predict_prob(text)
            if len(probs) < 2:
                print("⚠️ 모델 출력 오류\n")
                continue
            real_prob, fake_prob = probs[0], probs[1]

            keywords = crawler.extract_keywords(text)
            urls = crawler.search_news_urls(keywords)

            print(f"\n✅ 진짜뉴스 확률: {real_prob * 100:.2f}%")
            print(f"❌ 가짜뉴스 확률: {fake_prob * 100:.2f}%")
            print("🔑 키워드:", ", ".join(keywords))
            print("🔗 URL 목록:", urls if urls else "없음")

            docs = crawler.crawl_articles(urls)
            if not docs:
                print("⚠️ 관련 뉴스 없음\n")
                continue

            vector_store.build_index(docs)
            top_docs = vector_store.search(text, top_k=3)
            if not top_docs:
                print("⚠️ 유사한 뉴스 문서 없음\n")
                continue

            answer = generate_answer(text, top_docs)
            confidence = 0.8
            summaries = summarize_docs_korean(top_docs)

            print(f"\n💡 RAG 응답:\n{answer}")
            print(f"🔍 신뢰도: {confidence:.3f}")
            print("📝 요약:")
            for i, s in enumerate(summaries, 1):
                print(f"{i}. {s}")
            print("\n=====================\n")

        except Exception as e:
            print(f"⚠️ 오류 발생: {e}\n")

if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)).start()
    cli_loop()
