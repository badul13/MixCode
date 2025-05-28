import React from 'react';
import '../styles/App.css';

function Sidebar({ history }) {
  return (
    <div className="sidebar">
      <h2>판별 기록</h2>
      <ul>
        {history.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

export default Sidebar;