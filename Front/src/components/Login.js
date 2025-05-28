import React, { useState } from 'react';
import '../styles/Login.css';

function Login({ onLogin, goHome, goSignup }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async () => {
    setError('');
    if (!email || !password) {
      setError('아이디와 비밀번호를 입력하세요.');
      return;
    }

    try {
      const response = await fetch('https://your-api/login', {  // 백엔드 API주소 호출해야함
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.message || '로그인 실패');
      }
      
      onLogin();  
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="login-container">
      <h2>로그인</h2>
      <input
        type="text"
        placeholder="아이디"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
      />
      <input
        type="password"
        placeholder="비밀번호"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <div className="login-buttons">
        <button onClick={handleLogin}>로그인</button>
        <button onClick={goHome}>홈으로 이동</button>
      </div>
      {error && <p className="error">{error}</p>}
      <div className="Signup-link">
        <p>계정이 없으신가요? <button onClick={goSignup}>회원가입</button></p>
      </div>
    </div>
  );
}

export default Login;
