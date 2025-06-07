import React, { useState, useCallback, useEffect } from "react";
import "./styles/App.css";
import Sidebar from "./components/Sidebar";
import SearchBox from "./components/SearchBox";
import Chatbot from "./components/Chatbot";
import Login from "./components/Login";
import Signup from "./components/Signup";
import axiosInstance from "./api/axiosInstance";

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isLoginPage, setIsLoginPage] = useState(false);
  const [isSignupPage, setIsSignupPage] = useState(false);
  const [chatMode, setChatMode] = useState(false);
  const [chatQuery, setChatQuery] = useState("");
  const [history, setHistory] = useState([]);
  const [selectedHistory, setSelectedHistory] = useState(null);
  const [currentUserEmail, setCurrentUserEmail] = useState("");
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem("accessToken");
    const email = localStorage.getItem("userEmail");
    if (token && email) {
      setIsLoggedIn(true);
      setCurrentUserEmail(email);
      fetchHistory(token, 1);
    }
  }, []);

  const fetchHistory = async (token, pageNum = 1) => {
    try {
      const config = {
        headers: { Authorization: `Bearer ${token}` },
        params: { page: pageNum, limit: 10 },
      };
      const res = await axiosInstance.get(`/history`, config);
      const newData = res.data.data || [];
      setHistory((prev) => (pageNum === 1 ? newData : [...prev, ...newData]));
      setHasMore(newData.length === 10);
    } catch (err) {
      console.error("❌ 기록 불러오기 실패:", err);
      setHasMore(false);
    }
  };

  const loadMoreHistory = () => {
    if (!hasMore) return;
    const token = localStorage.getItem("accessToken");
    const nextPage = page + 1;
    setPage(nextPage);
    fetchHistory(token, nextPage);
  };

  const handleLogin = async (token, email) => {
<<<<<<< HEAD
    setIsLoginPage(false); // 🟢 먼저 false로 설정하여 렌더링 유도
=======
    setIsLoginPage(false);
>>>>>>> 31b39005 (기록 열람 추가..)
    localStorage.setItem("accessToken", token);
    localStorage.setItem("userEmail", email);
    setIsLoggedIn(true);
    setCurrentUserEmail(email);
    setPage(1);
    await fetchHistory(token, 1);
  };

  const handleLogout = () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("userEmail");
    setIsLoggedIn(false);
    setCurrentUserEmail("");
    setChatMode(false);
    setSelectedHistory(null);
    setHistory([]);
    setPage(1);
    setHasMore(true);
  };

  const goToLogin = () => {
    setIsLoginPage(true);
    setIsSignupPage(false);
    setChatMode(false);
  };

  const goToSignup = () => {
    setIsSignupPage(true);
    setIsLoginPage(false);
    setChatMode(false);
  };

  const goHome = () => {
    setIsLoginPage(false);
    setIsSignupPage(false);
    setChatMode(false);
    setSelectedHistory(null);
  };

  const handleSearch = (query) => {
    setSelectedHistory(null);
    setChatQuery(query);
    setChatMode(true);
  };

  const handleSelectHistory = async (item) => {
    try {
      const token = localStorage.getItem("accessToken");
      const res = await axiosInstance.get(`/news/historyContent/${item.id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setSelectedHistory(res.data.data);
      setChatMode(true);
    } catch (err) {
      console.error("❌ 검증 상세 조회 실패:", err);
    }
  };

  const updateCurrentHistory = useCallback(
    (chatMessages, title, timestamp) => {
      const newItem = {
        title: title || chatQuery,
        messages: chatMessages,
        timestamp: timestamp || new Date().toISOString(),
      };
      setHistory((prev) => [newItem, ...prev]);
    },
    [chatQuery]
  );

  if (isLoginPage) {
    return (
      <Login onLogin={handleLogin} goHome={goHome} goSignup={goToSignup} />
    );
  }

  if (isSignupPage) {
    return <Signup goHome={goHome} />;
  }

  return (
    <div className="app-container">
      <div className="header">
        <div className="logo" onClick={goHome}>
          Fake News Checker
        </div>
        <div className="login-button-wrapper">
          {!isLoggedIn ? (
            <button onClick={goToLogin}>로그인</button>
          ) : (
            <button onClick={handleLogout}>로그아웃</button>
          )}
        </div>
      </div>

      <div className="main-content">
        <Sidebar
          history={history}
          onSelect={handleSelectHistory}
          onLoadMore={loadMoreHistory}
        />
        {chatMode ? (
          <Chatbot
            query={chatQuery}
            goHome={goHome}
            initialMessages={selectedHistory?.messages || [{ sender: 'bot', text: selectedHistory?.summary }]}
            updateHistory={updateCurrentHistory}
          />
        ) : (
          <SearchBox onSearch={handleSearch} />
        )}
      </div>
    </div>
  );
}

export default App;