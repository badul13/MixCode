import os
import torch
import threading
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
load_dotenv()

app = Flask(__name__)
run_with_ngrok(app)
crawler = NaverNewsCrawler()

tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
bart_model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model.to(device)
bart_model.eval()

# 벡터 저장소
class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = None
        self.documents = []

    def build_index(self, docs: List[str]):
        self.documents = docs
        self.doc_vectors = self.vectorizer.fit_transform(docs) if docs else None

    def search(self, query: str, top_k=3) -> List[str]:
        if self.doc_vectors is None or not self.documents:
             return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [self.documents[i] for i in top_indices if similarities[i] > 0]


vector_store = VectorStore()

def summarize_docs_korean(docs: List[str], max_docs: int = 3) -> List[str]:
    summaries = []
    for doc in docs[:max_docs]:
        try:
            input_ids = tokenizer.encode(doc[:1024], return_tensors="pt", max_length=1024, truncation=True).to(device)
            summary_ids = bart_model.generate(
                input_ids, max_length=128, min_length=32,
                length_penalty=2.0, num_beams=4, early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"요약 실패: {e}")
    return summaries

# RAG 응답 
def generate_answer(question: str, contexts: List[str]) -> str:
    context_text = " ".join(contexts)
    input_text = f"{context_text} 질문: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
    summary_ids = bart_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 신뢰도 분리
def final_decision(real_prob, fake_prob, top_docs, answer):
    step1_confidence = max(real_prob, fake_prob)
    step2_confidence = min(len(top_docs) / 3, 1.0) if top_docs else 0.0

    if step1_confidence > 0.8 and step2_confidence > 0.6:
        decision = "🟢 진짜일 가능성이 높습니다."
    elif fake_prob > 0.8:
        decision = "🔴 가짜일 가능성이 높습니다."
    elif not top_docs:
        decision = "⚪ 관련 뉴스가 없어 진위를 판단하기 어렵습니다."
    else:
        decision = "🟡 확신할 수 없습니다. 추가 검토가 필요합니다."

    return decision, round(step1_confidence, 3), round(step2_confidence, 3)

# Flask API 
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

        vector_store.build_index(docs)
        top_docs = vector_store.search(text, top_k=3)
        summaries = summarize_docs_korean(top_docs)
        answer = generate_answer(text, top_docs) if top_docs else ""

        decision, step1_conf, step2_conf = final_decision(real_prob, fake_prob, top_docs, answer)

        return jsonify({
            "error": None,
            "real": round(real_prob, 3),
            "fake": round(fake_prob, 3),
            "step1_confidence": step1_conf,
            "step2_confidence": step2_conf,
            "confidence": max(step1_conf, step2_conf),
            "keywords": keywords,
            "urls": urls,
            "summaries": summaries,
            "ragAnswer": answer,
            "finalDecision": decision
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def wrap_text_by_sentences(text: str) -> str:
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    return "\n".join(sentences)

def wrap_text_by_length(text: str, max_len=80) -> str:
    lines = []
    while len(text) > max_len:
        split_pos = text.rfind(" ", 0, max_len)
        if split_pos == -1:
            split_pos = max_len
        lines.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    lines.append(text)
    return "\n".join(lines)

def format_summaries(summaries: List[str]) -> str:
    formatted = []
    for i, summary in enumerate(summaries, 1):
        wrapped = wrap_text_by_length(summary, max_len=60)
        formatted.append(f"{i}. \n{wrapped}")
    return "\n\n".join(formatted)

# CLI 
def cli_loop():

    print("뉴스 검색 챗봇 시작합니다. 종료하려면 'exit' 입력하세요.\n")
    while True:
        text = input(f"\033[33m문장을 입력해주세요: \033[0m").strip()
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

            print("🔑 핵심 키워드:", ", ".join(keywords))
            print("🔗 관련 기사 URL 목록:")
            if urls:
                for url in urls:
                    print(f"  - {url}")
            else:
                print("  없음")

            docs = crawler.crawl_articles(urls)
            if not docs:
                print("\n💡 RAG 응답:")
                print("정부의 발표와 관련된 뉴스는 확인되지 않았습니다.")
                continue

            vector_store.build_index(docs)
            top_docs = vector_store.search(text, top_k=3)
            if not top_docs:
                print("\n💡 RAG 응답:")
                print("관련 뉴스가 충분하지 않습니다.")
                continue

            answer = generate_answer(text, top_docs)
            summaries = summarize_docs_korean(top_docs)

            # RAG 기반 신뢰도 (예시로 고정값 사용)
            rag_confidence = len(top_docs) / 4

            print("\n💡 RAG 응답:")
            print(format_summaries(summaries))
            
            print("\n📝 요약:")
            for i, s in enumerate(summaries, 1):
                print(f"{i}. {wrap_text_by_length(s)}")

            print("\n" + "─" * 70)

            print(f"\n🔍 1단계 신뢰도 (모델 기반): {real_prob:.3f}")
            print(f"🔎 2단계 신뢰도 (RAG 기반): {rag_confidence:.3f}")

            print(f"\n✅ 진짜뉴스 확률: {real_prob * 100:.2f}%")
            print(f"❌ 가짜뉴스 확률: {fake_prob * 100:.2f}%")

            final_score = real_prob * 0.5 + rag_confidence * 0.5
            final_label = "✅ 진짜일 가능성이 높은 기사입니다!" if final_score >= 0.5 else "❌ 가짜일 가능성이 높은 기사입니다!"

            print(f"\n📌 최종 판단: {final_label}\n")
            print("─" * 70 + "\n")

        except Exception as e:
            print(f"⚠️ 오류 발생: {e}\n")

# 실행
if __name__ == "__main__":
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)).start()
    cli_loop()
