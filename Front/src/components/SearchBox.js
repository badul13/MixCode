import React, { useState } from 'react';
import '../styles/SearchBox.css';

const SearchBox = ({ onSearch }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query);
      setQuery('');
    }
  };

  return (
    <div className="search-box">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="판별할 뉴스를 입력하세요"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit">↑</button>
      </form>
    </div>
  );
};

export default SearchBox;