import React, { useRef, useEffect } from 'react';
import '../styles/App.css';

function Sidebar({ history, onSelect, onLoadMore }) {
  const listRef = useRef(null);

  // 스크롤 이벤트 핸들러
  const handleScroll = () => {
    const container = listRef.current;
    if (!container) return;

    if (container.scrollTop + container.clientHeight >= container.scrollHeight - 10) {
      onLoadMore(); // 하단에 도달하면 추가 데이터 요청
    }
  };

  useEffect(() => {
    const container = listRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
    }

    return () => {
      if (container) {
        container.removeEventListener('scroll', handleScroll);
      }
    };
  }, []);

  return (
    <div className="sidebar" ref={listRef}>
      <h2>판별 기록</h2>
      <ul>
        {history.map((item, index) => (
          <li key={index} onClick={() => onSelect(item)}>
            {item.query}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Sidebar;