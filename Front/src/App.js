import React, { useState } from 'react';
import './styles/App.css';
import Sidebar from './components/Sidebar';
import SearchBox from './components/SearchBox';
import Chatbot from './components/Chatbot';
import Login from './components/Login';
import Signup from './components/Signup';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isLoginPage, setIsLoginPage] = useState(false);
  const [isSignupPage, setIsSignupPage] = useState(false);
  const [chatMode, setChatMode] = useState(false);
  const [chatQuery, setChatQuery] = useState('');
  const [history, setHistory] = useState([]);

  const handleLogin = () => {
    setIsLoggedIn(true);
    setIsLoginPage(false);
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
  };

  const handleSearch = (query) => {
    setChatQuery(query);
    setHistory([query, ...history]);
    setChatMode(true);
  };

  if (isLoginPage) {
    return <Login onLogin={handleLogin} goHome={goHome} goSignup={goToSignup} />;
  }

  if (isSignupPage) {
    return <Signup goHome={goHome} />;
  }

  return (
    <div className="app-container">
      <div className="header">
        <div className="logo" onClick={goHome}>Fake News Checker</div>
        <div className="login-button-wrapper">
          {!isLoggedIn && <button onClick={goToLogin}>로그인</button>}
        </div>
      </div>

      <div className="main-content">
        {chatMode ? (
          <Chatbot query={chatQuery} goHome={goHome} />
        ) : (
          <>
            <Sidebar history={history} />
            <SearchBox onSearch={handleSearch} />
          </>
        )}
      </div>
    </div>
  );
}

export default App;
