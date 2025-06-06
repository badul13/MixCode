import React, { useEffect, useRef } from 'react';
import '../styles/App.css';

function Sidebar({ history, onSelect, onLoadMore }) {
  const listRef = useRef(null);

  useEffect(() => {
    const listEl = listRef.current;
    if (!listEl) return;

    const handleScroll = () => {
      if (listEl.scrollTop + listEl.clientHeight >= listEl.scrollHeight - 10) {
        onLoadMore();
      }
    };

    listEl.addEventListener('scroll', handleScroll);
    return () => listEl.removeEventListener('scroll', handleScroll);
  }, [onLoadMore]);

  return (
    <div className="sidebar" ref={listRef}>
      <h2>판별 기록</h2>
      <ul>
        {history.map((item, index) => (
          <li key={index} onClick={() => onSelect(item)}>
            {item.title || item.query}
            </li>
        ))}
      </ul>
    </div>
  );
}

export default Sidebar;
