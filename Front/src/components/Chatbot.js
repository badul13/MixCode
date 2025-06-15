// Chatbot.js
import React, { useEffect, useState, useRef } from "react";
import "../styles/Chatbot.css";

const token = localStorage.getItem("accessToken");

// JSON 데이터를 자연어 문장으로 변환하는 함수
function convertStructuredJsonToNatural(data) {
  const labelText =
    data.label === 0
      ? "🟢 이 뉴스는 **진짜 뉴스**로 판단됩니다."
      : "🔴 이 뉴스는 **가짜 뉴스**로 판단됩니다.";

  const labelTitle = data.label === 0 ? "🟢 진짜 뉴스" : "🔴 가짜 뉴스";

  return `${labelTitle}
  
📰 뉴스 제목:
${data.newsTitle}

🧾 뉴스 본문:
${data.text}

📌 요약:
${data.summary}

🧠 RAG 응답:
${data.ragAnswer}

📅 분석 시간: ${data.timestamp}

${labelText}`;
}

// 챗봇 컴포넌트
function Chatbot({ query, goHome }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const chatEndRef = useRef(null);

<<<<<<< HEAD
  const fetchResponse = useCallback(
    async (q) => {
      try {
        const res = await axiosInstance.post("/news/chat", { message: q });
        const { title, chat, timestamp } = res.data;

        setMessages((prev) => [...prev, ...chat]);
        updateHistory(chat, title, timestamp);
      } catch (err) {
        console.error("❌ 서버 응답 오류:", err);
        setMessages((prev) => [
          ...prev,
          { sender: "bot", text: "⚠️ 서버와 연결할 수 없습니다." },
        ]);
      }
    },
    [updateHistory]
  );

  useLayoutEffect(() => {
    if (initialMessages && initialMessages.length > 0) {
      setMessages(initialMessages);
    } else if (query) {
      const userMessage = { sender: "user", text: query };
      setMessages([userMessage]);
    } else {
      setMessages([]);
    }
  }, [query, initialMessages]);

=======
  // 초기 query 입력 시 자동 응답
>>>>>>> 0ab6ba0fbdb96b053702e4d0f16f5f35349f0361
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
      const response = await fetch(
        "https://76c9-175-115-149-2.ngrok-free.app/news/chat",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: token,
          },
          body: JSON.stringify({ text: q }),
        }
      );

      if (!response.ok) throw new Error("백엔드 응답 오류");

      const responseData = await response.json();
      const data = responseData.data; // ⚠️ 여기서 .data를 꺼내야 함
      const natural = convertStructuredJsonToNatural(data);

      setMessages((prev) => [...prev, { sender: "bot", text: natural }]);
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
      <div className="chatbot-messages" style={{ whiteSpace: "pre-line" }}>
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
