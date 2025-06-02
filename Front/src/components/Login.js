import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import axiosInstance from "../api/axiosInstance";
import "../styles/Login.css";

function Login({ onLogin, goHome, goSignup }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async () => {
    setError("");
    if (!email || !password) {
      setError("아이디와 비밀번호를 입력하세요.");
      return;
    }

    try {
      const response = await axiosInstance.post(`/user/signin`, {
        email,
        password,
      });

      const { success, data } = response.data;

      if (success && data) {
        const { accessToken, refreshToken } = data;

        // ✅ 콘솔에서 확인
        console.log("✅ accessToken:", accessToken);
        console.log("✅ refreshToken:", refreshToken);

        // ✅ localStorage에 저장
        localStorage.setItem("accessToken", accessToken);
        localStorage.setItem("refreshToken", refreshToken);
        localStorage.setItem("userEmail", email);

        // ✅ 저장 확인 로그
        console.log(
          "🧪 저장된 accessToken:",
          localStorage.getItem("accessToken")
        );

        // ✅ 약간의 지연 후 페이지 이동
        setTimeout(() => {
          navigate("/chatbot");
        }, 200);
      } else {
        alert("❌ 로그인 실패: 서버 응답 이상");
      }
    } catch (error) {
      console.error("로그인 에러:", error);
      alert("❌ 로그인 중 오류 발생");
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
        <p>
          계정이 없으신가요? <button onClick={goSignup}>회원가입</button>
        </p>
      </div>
    </div>
  );
}

export default Login;
