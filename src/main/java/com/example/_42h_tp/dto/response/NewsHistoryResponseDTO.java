package com.example._42h_tp.dto.response;

import com.example._42h_tp.dto.request.NewsVerifyRequestDTO;
import lombok.Builder;
import lombok.Getter;

import java.util.List;

@Getter
@Builder
public class NewsHistoryResponseDTO {
    private List<NewsVerificationDTO> verifications;
    private PageInfo pageInfo;

    @Getter
    @Builder
    public static class PageInfo {
        private int currentPage;
        private int totalPages;
        private long totalElements;
        private boolean hasNext;
    }
} 