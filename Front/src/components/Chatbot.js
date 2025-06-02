import React, { useLayoutEffect, useEffect, useState, useRef, useCallback } from 'react';
import '../styles/Chatbot.css';
import axiosInstance from '../api/axiosInstance';

function Chatbot({ query, goHome, initialMessages = null, updateHistory }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const chatEndRef = useRef(null);

  const fetchResponse = useCallback(async (q) => {
    try {
<<<<<<< HEAD
      const response = await fetch(
        "https://4dea-210-119-104-214.ngrok-free.app/news/chat",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            // Authorization: `Bearer ${token}`, // 여기서 사용자 인증됨
          },
          body: JSON.stringify({ message: q }),
        }
      );
=======
      const res = await axiosInstance.post('/chat', { message: q });
      const { title, chat, timestamp } = res.data;
>>>>>>> 67aa30ce (api호출 추가)

      setMessages((prev) => [...prev, ...chat]);
      updateHistory(chat, title, timestamp);
    } catch (err) {
      console.error('❌ 서버 응답 오류:', err);
      setMessages((prev) => [
        ...prev,
        { sender: 'bot', text: '⚠️ 서버와 연결할 수 없습니다.' },
      ]);
    }
  }, [updateHistory]);

  useLayoutEffect(() => {
    if (initialMessages && initialMessages.length > 0) {
      setMessages(initialMessages);
    } else if (query) {
      const userMessage = { sender: 'user', text: query };
      setMessages([userMessage]);
    } else {
      setMessages([]);
    }
  }, [query, initialMessages]);

  useEffect(() => {
    if (!initialMessages && query) {
      fetchResponse(query);
    }
  }, [query, initialMessages, fetchResponse]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const userMessage = { sender: 'user', text: input.trim() };
    setMessages((prev) => [...prev, userMessage]);
    fetchResponse(input.trim());
    setInput('');
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
