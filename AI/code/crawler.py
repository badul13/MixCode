import os
import torch
import threading
import pandas as pd
from flask import Flask, request, jsonify
from typing import List
from dotenv import load_dotenv
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, BertTokenizer, BertForSequenceClassification, AutoTokenizer
from naver_news_crawler import NaverNewsCrawler
from pyngrok import ngrok
from flask_cors import CORS
from rag_engine import RAGEngine
import numpy as np
import datetime

# 환경변수 로드
load_dotenv()

# ngrok 토큰 환경변수에서 불러오기

load_dotenv()

ngrok.set_auth_token("2xwNdP6n13UgVoHisO4aafDb8kq_2J1uW4qGbkwE3CbDztGKn")

# RAG 엔진 인스턴스 생성
vector_store = RAGEngine()

# 데이터 로드 및 인덱스 빌드 (초기)
df = pd.read_csv('outputdata.csv')
docs = df['text'].dropna().tolist()
vector_store.build_index(docs)

app = Flask(__name__)
CORS(app)

crawler = NaverNewsCrawler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 요약 모델 로드 (KoBART)
tokenizer_kobart = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
model_kobart = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")
model_kobart.to(device)
model_kobart.eval()

# 1단계 KcELECTRA 기반 가짜 뉴스 분류기
class KcElectraClassifier:
    def __init__(self, model_path="news_classifier_model.pt"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
        self.model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=2)
        
        # 이 줄 추가: 사전 학습된 모델 파라미터 로드
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

    def predict_prob(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        # (진짜:1, 가짜:0)
        return probs[1], probs[0]

# 2단계 요약 기반 KoBERT 분류기 (RAG용)
class RagKoBertClassifier:
    def __init__(self, model_path="rag_classifier_model.pt"):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
        self.model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=2)
        
        # 가중치 불러오기 추가
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

    def predict_rag_prob(self, text, summary_concat):
        input_text = text + " [SEP] " + summary_concat
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        label = int(probs.argmax())
        return probs[1], probs[0], label

# 요약 함수
def summarize_docs_korean(docs: List[str], max_docs: int = 3) -> List[str]:
    summaries = []
    for doc in docs[:max_docs]:
        try:
            input_ids = tokenizer_kobart.encode(doc[:1024], return_tensors="pt", max_length=1024, truncation=True).to(device)
            summary_ids = model_kobart.generate(
                input_ids,
                max_length=128,
                min_length=64,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
            summary = tokenizer_kobart.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"요약 실패: {e}")
    return summaries

def clean_summary(text: str) -> str:
    awkward_starts = ["했다.", "이다.", "요?"]
    text = text.strip()
    for aw in awkward_starts:
        if text.startswith(aw):
            text = text[len(aw):].strip()
    if text and not text.endswith('.'):
        text += '.'
    return text

# 1단계, 2단계 확률 합치는 함수 (가중치 적용 예시)
def combined_prob(real1, fake1, real2, fake2, weight=0.5):
    combined_real = real1 * (1 - weight) + real2 * weight
    combined_fake = fake1 * (1 - weight) + fake2 * weight
    s = combined_real + combined_fake
    if s == 0:
        return 0.0, 0.0
    return combined_real / s, combined_fake / s

# 모델 초기화 (체크포인트 경로 포함)
kc_electra_model = KcElectraClassifier("news_classifier_model.pt")
rag_kobert_model = RagKoBertClassifier("rag_classifier_model.pt")

def generate_answer(query: str, docs: List[str]):
    answer = " ".join(docs[:3])
    confidence = 0.9  # 필요시 수정
    return answer, confidence

def convert_numpy(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    else:
        return obj

@app.route("/verify", methods=["POST"])
def verify_news():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "'text' 필드가 요청에 포함되어야 합니다."}), 400

        text = data["text"]

        # 1단계 예측
        real_prob_1, fake_prob_1 = kc_electra_model.predict_prob(text)

        # 키워드 추출 및 관련 뉴스 크롤링
        keywords = crawler.extract_keywords(text)
        urls = crawler.search_news_urls(keywords)
        docs = crawler.crawl_articles(urls)

        # RAG 인덱스 재빌드 및 검색
        vector_store.build_index(docs)
        top_docs = vector_store.search(text, top_k=3)

        # RAG 답변 생성 및 요약
        answer, rag_confidence = generate_answer(text, top_docs)
        summaries = summarize_docs_korean(top_docs)
        summary_concat = " ".join(summaries)

        # 2단계 RAG 기반 분류
        rag_real_prob, rag_fake_prob, rag_label = rag_kobert_model.predict_rag_prob(text, summary_concat)

        # 뉴스 제목 (urls에서 실제 제목을 얻으려면 크롤러 수정 필요)
        title = urls[0] if urls else ""
        content = top_docs[0] if top_docs else ""
        summary_text = summaries[0] if summaries else ""

        response_data = {
            "query": text,
            "news_title": title,
            "text": content,
            "summary": summary_text,
            "label": int(fake_prob_1 > 0.5),
            "fake_prob": str(float(fake_prob_1)),
            "rag_prob": str(float(rag_fake_prob)),
            "real_prob_percent": round((1 - float(fake_prob_1)) * 100, 2),
            "fake_prob_percent": round(float(fake_prob_1) * 100, 2),
            "rag_answer": answer,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # numpy 타입 JSON 직렬화 대응
        response_data = convert_numpy(response_data)

        return jsonify(response_data)

    except Exception as e:
        print(f"verify_news error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/ngrok_url")
def get_ngrok_url():
    return jsonify({"ngrok_url": public_url})

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

def final_decision(real_prob_1, fake_prob_1, rag_real_prob, rag_fake_prob, threshold=0.5):
    combined_real, combined_fake = combined_prob(real_prob_1, fake_prob_1, rag_real_prob, rag_fake_prob)
    if combined_real >= threshold:
        return 1, combined_real
    else:
        return 0, combined_fake

def cli_loop():
    print("뉴스 검색 챗봇 시작합니다! 종료하려면 'exit' 입력하세요.\n")
    while True:
        text = input("문장을 입력하세요: ")
        if text.lower() == "exit":
            print("챗봇을 종료합니다.")
            break
        try:
            real_prob_1, fake_prob_1 = kc_electra_model.predict_prob(text)
            keywords = crawler.extract_keywords(text)
            urls = crawler.search_news_urls(keywords)
            docs = crawler.crawl_articles(urls)

            if not docs:
                print("⚠️ 관련 뉴스 없음\n")
                continue

            vector_store.build_index(docs)
            top_docs = vector_store.search(text, top_k=3)

            answer, rag_confidence = generate_answer(text, top_docs)
            summaries = summarize_docs_korean(top_docs)
            summary_concat = " ".join(summaries)

            rag_real_prob, rag_fake_prob, rag_label = rag_kobert_model.predict_rag_prob(text, summary_concat)

            combined_real, combined_fake = combined_prob(
                real_prob_1, fake_prob_1, rag_real_prob, rag_fake_prob, weight=0.5
            )

            print("🔑 핵심 키워드:", ", ".join(keywords))
            print("🔗 관련 기사 URL 목록:", urls)

            print(f"\n💡 RAG 응답:\n{answer}")

            print("📝 요약:\n")
            for i, summary in enumerate(summaries, 1):
                s_clean = clean_summary(summary)
                if s_clean:
                    print(f"{i}. {s_clean}\n")

            print("🔍 신뢰도 분석 결과:")
            print(f"- KoBERT 분류 결과: {'가짜뉴스' if fake_prob_1 >= 0.5 else '진짜뉴스'} ({fake_prob_1:.3f}%)")
            print(f"- RAG 기반 분류 결과: {'가짜뉴스' if rag_fake_prob >= 0.5 else '진짜뉴스'} ({rag_confidence:.3f}%)\n")

            print(f"\n⭕ 진짜뉴스 확률: {combined_real * 100:.2f}%")
            print(f"❌ 가짜뉴스 확률: {combined_fake * 100:.2f}%")

            final_label, final_conf = final_decision(real_prob_1, fake_prob_1, rag_real_prob, rag_fake_prob)
            label_text = "진짜뉴스" if final_label == 1 else "가짜뉴스"
            print(f"\n▶ 최종 판단: {label_text} (신뢰도 {final_conf * 100:.2f}%)\n")

        except Exception as e:
            print(f"오류 발생: {e}\n")

if __name__ == "__main__":
    # ngrok 터널 시작
    public_url = ngrok.connect(5000).public_url
    print(f" * ngrok URL: {public_url}")

    # Flask 서버 별도 스레드에서 실행
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)).start()

    # CLI 루프 시작
    cli_loop()
