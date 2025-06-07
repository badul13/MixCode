import axios from "axios";

const baseURL = "https://76c9-175-115-149-2.ngrok-free.app";

const axiosInstance = axios.create({ baseURL });

// ✅ interceptor는 실행 시마다 최신 토큰을 불러오게 해야 함
axiosInstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("accessToken"); // 이 줄이 요청마다 실행되도록 유지
    console.log("요청 전 토큰:", token);
    if (token) {
      config.headers.Authorization = token;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export default axiosInstance;
