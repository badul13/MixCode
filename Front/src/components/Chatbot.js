// Chatbot.js
import React, { useEffect, useState, useRef } from 'react';
import '../styles/Chatbot.css';

function Chatbot({ query, goHome }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (query) {
      const userMessage = { sender: 'user', text: query };
      setMessages((prev) => [...prev, userMessage]);
      fetchResponse(query);
    }
  }, [query]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const fetchResponse = async (q) => { //여기에 API 호출 
    const botMessage = {
      sender: 'bot',
      text: `"${q}"에 대한 응답입니다.`
    };
    setMessages((prev) => [...prev, botMessage]);
  };

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


      {/* </div>const fetchResponse = async (q) => { 
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
  }; */}


