import React, { useRef, useEffect } from 'react';
import '../styles/App.css';

function Sidebar({ history, onSelect, onLoadMore }) {
  const listRef = useRef(null);

  const handleScroll = () => {
    const { scrollTop, scrollHeight, clientHeight } = listRef.current;
    if (scrollTop + clientHeight >= scrollHeight - 10) {
      onLoadMore();
    }
  };

  useEffect(() => {
    const ref = listRef.current;
    if (ref) {
      ref.addEventListener('scroll', handleScroll);
      return () => ref.removeEventListener('scroll', handleScroll);
    }
  }, []);

  return (
    <div className="sidebar" ref={listRef}>
      <h2>판별 기록</h2>
      <ul>
        {history.map((item, index) => (
          <li key={index} onClick={() => onSelect(item.id)}>
            {item.title}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Sidebar;