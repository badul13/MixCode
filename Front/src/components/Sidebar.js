import React from "react";
import "../styles/App.css";

function Sidebar({ history, onSelect, onLoadMore }) {
  const handleScroll = (e) => {
    const bottom =
      e.target.scrollHeight - e.target.scrollTop === e.target.clientHeight;
    if (bottom && onLoadMore) onLoadMore();
  };

  return (
    <div className="sidebar" onScroll={handleScroll}>
      <h2>판별 기록</h2>
      <ul>
        {history.map((item) => (
          <li key={item.id} onClick={() => onSelect(item)}>
            {item.title}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Sidebar;
