// Chatbot.js
import React, { useEffect, useState, useRef } from "react";
import "../styles/Chatbot.css";

// 텍스트를 JSON으로 안전하게 파싱하는 함수
function safeParseTextToJson(text) {
  try {
    const keywords =
      text
        .match(/• 핵심 키워드: (.+)/)?.[1]
        .split(",")
        .map((s) => s.trim()) || [];

    const urls =
      [...text.matchAll(/- (https?:\/\/\S+)/g)].map((m) => m[1]) || [];

    const rag =
      text.match(/RAG 응답:\n([\s\S]*?)\n\n?요약:/)?.[1]?.trim() || "";

    const summary =
      text.match(/요약:\n([\s\S]*?)\n\n?(1단계|신뢰도)/)?.[1]?.trim() || "";

    const score_model =
      parseFloat(text.match(/신뢰도 \(모델 기반\): (\d+\.\d+)/)?.[1]) || 0;
    const score_rag =
      parseFloat(text.match(/신뢰도 \(RAG 기반\) ?: (\d+\.\d+)/)?.[1]) || 0;

    const realProb =
      parseFloat(text.match(/진짜뉴스 확률: (\d+\.\d+)/)?.[1]) / 100 || 0;
    const fakeProb =
      parseFloat(text.match(/가짜뉴스 확률: (\d+\.\d+)/)?.[1]) / 100 || 0;

    const conclusion = text.match(/최종 판단:\n(.+)/)?.[1]?.trim() || "";

    return {
      keywords,
      urls,
      rag,
      summary,
      score_model,
      score_rag,
      real_prob: realProb,
      fake_prob: fakeProb,
      conclusion,
    };
  } catch (e) {
    console.error("파싱 실패:", e);
    return {
      keywords: [],
      urls: [],
      rag: "",
      summary: "",
      score_model: 0,
      score_rag: 0,
      real_prob: 0,
      fake_prob: 0,
      conclusion: "분석 실패",
    };
  }
}

// JSON 데이터를 자연어로 변환하는 함수
function convertJsonToNatural(data) {
  const real = (data.real_prob * 100).toFixed(2);
  const fake = (data.fake_prob * 100).toFixed(2);
  const keywords = data.keywords.join(", ");
  const urls = data.urls.map((url) => `- ${url}`).join("\n");

  return `🔍 핵심 키워드: ${keywords}

📎 관련 기사:
${urls}

🧠 RAG 응답:
${data.rag}

📝 요약:
${data.summary}

📊 신뢰도:
- 모델 기반: ${data.score_model.toFixed(3)}
- RAG 기반: ${data.score_rag.toFixed(3)}

✅ 진짜뉴스 확률: ${real}%
❌ 가짜뉴스 확률: ${fake}%

📌 최종 판단:
${data.conclusion}`;
}

// 챗봇 컴포넌트
function Chatbot({ query, goHome }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (query) {
      const userMessage = { sender: "user", text: query };
      setMessages((prev) => [...prev, userMessage]);
      fetchResponse(query);
    }
  }, [query]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchResponse = async (q) => {
    try {
      const response = await fetch("https://your-ngrok-id.ngrok.io/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q }),
      });

      if (!response.ok) throw new Error("백엔드 응답 오류");

      const text = await response.text();
      const parsed = safeParseTextToJson(text);
      const natural = convertJsonToNatural(parsed);

      // 사용자 출력용 메시지
      setMessages((prev) => [...prev, { sender: "bot", text: natural }]);

      // 🔽 여기서 parsed JSON을 DB 저장용으로 백엔드에 전송
      await saveParsedJsonToBackend(parsed);
    } catch (error) {
      console.error("API 호출 오류:", error);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "⚠️ 서버와 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.",
        },
      ]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userMessage = { sender: "user", text: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    fetchResponse(input.trim());
    setInput("");
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <button onClick={goHome}>← 홈으로</button>
        <h2>챗봇 응답</h2>
      </div>
      <div className="chatbot-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <div className="bubble">{msg.text}</div>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>
      <form className="chatbot-input" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="질문을 입력하세요"
        />
        <button type="submit">→</button>
      </form>
    </div>
  );
}

export default Chatbot;

{
  /* </div>const fetchResponse = async (q) => { 
    try {
      const response = await fetch('https://your-chat-api.com/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // 필요한 경우 Authorization 헤더도 추가
        },
        body: JSON.stringify({ question: q })
      });
  
      const data = await response.json();
  
      const botMessage = {
        sender: 'bot',
        text: data.answer || '응답을 받지 못했습니다.'
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('API 호출 오류:', error);
      const botMessage = {
        sender: 'bot',
        text: '오류가 발생했습니다. 다시 시도해주세요.'
      };
      setMessages((prev) => [...prev, botMessage]);
    }
  }; */
}
