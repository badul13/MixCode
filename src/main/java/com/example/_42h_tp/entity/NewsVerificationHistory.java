package com.example._42h_tp.entity;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;

import java.time.LocalDateTime;

@Entity
@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NewsVerificationHistory {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(nullable = false)
    private String query;

    @Column(nullable = false)
    private String newsTitle;

    @Column(nullable = false, length = 5000)
    private String newsContent;

    @Column(nullable = false, length = 5000)
    private String summary;

    @Column(nullable = false)
    private Integer label;

    @Column(nullable = false)
    private Double fakeProb;

    @Column(nullable = false)
    private Double ragProb;

    @Column(nullable = false)
    private Double realProbPercent;

    @Column(nullable = false)
    private Double fakeProbPercent;

    @Column(nullable = false, length = 5000)
    private String ragAnswer;

    @Column(nullable = false, updatable = false)
    private String timestamp;
}
