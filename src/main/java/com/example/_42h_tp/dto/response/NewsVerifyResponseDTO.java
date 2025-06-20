package com.example._42h_tp.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class NewsVerifyResponseDTO {
    private String query;
    private String newsTitle;
    private String text;
    private String summary;
    private Integer label;
    private Double fakeProb;
    private Double ragProb;
    private Double realProbPercent;
    private Double fakeProbPercent;
    private String ragAnswer;
    private String timestamp;
}