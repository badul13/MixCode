package com.example._42h_tp.repository;

import com.example._42h_tp.entity.NewsVerificationHistory;
import com.example._42h_tp.entity.User;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface NewsVerificationHistoryRepository extends JpaRepository<NewsVerificationHistory, Long> {
    Page<NewsVerificationHistory> findByUserOrderByIdDesc(User user, Pageable pageable);
} 