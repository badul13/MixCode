import React, { useState } from 'react';
import axiosInstance from '../api/axiosInstance';
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
      const response = await axiosInstance.post(`/user/signin`, {
        email,
        password,
      });

      const { success, data, message } = response.data;

      if (!success) {
        throw new Error(message || '로그인 실패');
      }

      // ✅ 토큰 저장
      localStorage.setItem('accessToken', data.accessToken);
      localStorage.setItem('refreshToken', data.refreshToken);

      console.log('✅ 로그인 성공');
      onLogin(); // App 컴포넌트의 로그인 상태 변경
    } catch (err) {
      setError(err.response?.data?.message || err.message || '로그인 실패');
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