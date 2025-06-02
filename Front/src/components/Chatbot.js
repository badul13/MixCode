// Chatbot.js
import React, { useEffect, useState, useRef } from "react";
import "../styles/Chatbot.css";

// JSON 데이터를 자연어 문장으로 변환하는 함수
function convertStructuredJsonToNatural(data) {
  const labelText =
    data.label === 0
      ? "🟢 이 뉴스는 **진짜 뉴스**로 판단됩니다."
      : "🔴 이 뉴스는 **가짜 뉴스**로 판단됩니다.";

  return `📰 뉴스 제목:
${data.news_title}

🧾 뉴스 본문:
${data.test}

📌 요약:
${data.summary}

🧠 RAG 응답:
${data.rag_answer}

📅 분석 시간: ${data.timestamp}

${labelText}`;
}

// DB 저장 요청 함수
async function saveParsedJsonToBackend(data) {
  try {
    const response = await fetch("https://your-ngrok-id.ngrok.io/save-result", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error("DB 저장 실패");
    }

    console.log("✅ DB에 저장 완료");
  } catch (error) {
    console.error("❌ DB 저장 오류:", error);
  }
}

// 챗봇 컴포넌트
function Chatbot({ query, goHome }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const chatEndRef = useRef(null);

  // 초기 query 입력 시 자동 응답
  useEffect(() => {
    if (query) {
      const userMessage = { sender: "user", text: query };
      setMessages((prev) => [...prev, userMessage]);
      fetchResponse(query);
    }
  }, [query]);

  // 메시지 추가 시 자동 스크롤
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // AI 응답 요청 + 출력 + 저장
  const fetchResponse = async (q) => {
    try {
      const response = await fetch("https://your-ngrok-id.ngrok.io/news/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: q }),
      });

      if (!response.ok) throw new Error("백엔드 응답 오류");

      const data = await response.json();
      const natural = convertStructuredJsonToNatural(data);

      setMessages((prev) => [...prev, { sender: "bot", text: natural }]);

      await saveParsedJsonToBackend(data);
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

  // 사용자 입력 처리
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
