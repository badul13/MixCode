// Chatbot.js
import React, { useEffect, useState, useRef } from "react";
import "../styles/Chatbot.css";

// 텍스트를 JSON으로 안전하게 파싱하는 함수
function safeParseTextToJson(text) {
  try {
    const realProb =
      parseFloat(text.match(/진짜뉴스 확률: (\d+\.\d+)%/)?.[1]) / 100 || 0;
    const fakeProb =
      parseFloat(text.match(/가짜뉴스 확률: (\d+\.\d+)%/)?.[1]) / 100 || 0;
    const keywords =
      text
        .match(/🔑 주요 키워드:\n([\s\S]*?)\n\n/)?.[1]
        .split(",")
        .map((s) => s.trim()) || [];
    const urls =
      [...text.matchAll(/- (https?:\/\/\S+)/g)].map((m) => m[1]) || [];
    const rag = text.match(/💡 RAG 답변:\n([\s\S]*?)\n\n🔍/)?.[1]?.trim() || "";
    const score = parseFloat(text.match(/신뢰도 점수: (\d+\.\d+)/)?.[1]) || 0;

    return {
      real_prob: realProb,
      fake_prob: fakeProb,
      keywords,
      urls,
      rag,
      score,
    };
  } catch (e) {
    console.error("파싱 실패:", e);
    return {
      real_prob: 0,
      fake_prob: 0,
      keywords: [],
      urls: [],
      rag: "분석 실패",
      score: 0,
    };
  }
}

// JSON 데이터를 자연어로 변환하는 함수
function convertJsonToNatural(data) {
  const real = (data.real_prob * 100).toFixed(2);
  const fake = (data.fake_prob * 100).toFixed(2);
  const keywords = data.keywords.join(", ");
  const urls = data.urls.map((url) => `- ${url}`).join("\n");

  const isReal = data.real_prob > data.fake_prob;

  const headline = isReal
    ? "이 뉴스는 진짜일 가능성이 높아요."
    : "이 뉴스는 가짜일 가능성이 높아요.";

  return `${headline}\n\n
  🔎 분석 키워드는 "${keywords}"이며,\n
  진짜뉴스 확률은 ${real}%, 가짜뉴스 확률은 **${fake}%**로 나타났습니다.\n\n
  🧠 관련 기사 3건의 요약:\n
  ${data.rag}\n\n
  📎 관련 기사 링크:\n${urls}\n\n신뢰도 점수는 ${data.score.toFixed(
    3
  )}입니다. ${isReal ? "참고하실 만한 정보입니다." : "주의가 필요합니다."}`;
}

// 백엔드에 파싱된 JSON 데이터를 저장하는 함수
async function saveParsedJsonToBackend(data) {
  try {
    const response = await fetch("https://your-ngrok-id.ngrok.io/save-result", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data), // parsed JSON 그대로 전송
    });

    if (!response.ok) {
      throw new Error("결과 저장 실패");
    }

    console.log("✅ 분석 결과가 성공적으로 저장되었습니다.");
  } catch (error) {
    console.error("❌ 결과 저장 오류:", error);
  }
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
