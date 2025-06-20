package com.example._42h_tp.service;

import com.example._42h_tp.GeneralException;
import com.example._42h_tp.config.WebClientConfig;
import com.example._42h_tp.constant.ErrorInfo;
import com.example._42h_tp.dto.request.NewsHistoryRequestDTO;
import com.example._42h_tp.dto.request.NewsVerifyRequestDTO;
import com.example._42h_tp.dto.response.NewsHistoryContentResponseDTO;
import com.example._42h_tp.dto.response.NewsHistoryResponseDTO;
import com.example._42h_tp.dto.response.NewsVerificationDTO;
import com.example._42h_tp.dto.response.NewsVerifyResponseDTO;
import com.example._42h_tp.entity.NewsVerificationHistory;
import com.example._42h_tp.entity.User;
import com.example._42h_tp.repository.NewsVerificationHistoryRepository;
import com.example._42h_tp.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.core.publisher.Mono;

@Service
@Slf4j
@RequiredArgsConstructor
public class NewsService {

    private final WebClient aiServerWebClient;
    private final NewsVerificationHistoryRepository newsVerificationHistoryRepository;
    private final UserRepository userRepository;

    public NewsVerifyResponseDTO verify(Long userId, NewsVerifyRequestDTO request) {

        log.error("@@@@@@@@@@@222");
        log.error(request.getText());

        try {
            NewsVerifyResponseDTO response = aiServerWebClient.post()
                    .uri("/verify")
                    .bodyValue(request)
                    .retrieve()
                    .bodyToMono(NewsVerifyResponseDTO.class)
                    .block();

            User user = userRepository.findById(userId)
                    .orElseThrow(() -> new GeneralException(ErrorInfo.USER_NOT_FOUND));

            NewsVerificationHistory history = NewsVerificationHistory.builder()
                    .user(user)
                    .query(response.getQuery())
                    .newsTitle(response.getNewsTitle())
                    .newsContent(response.getText())
                    .summary(response.getSummary())
                    .label(response.getLabel())
                    .fakeProb(response.getFakeProb())
                    .ragProb(response.getRagProb())
                    .realProbPercent(response.getRealProbPercent())
                    .fakeProbPercent(response.getFakeProbPercent())
                    .ragAnswer(response.getRagAnswer())
                    .timestamp(response.getTimestamp())
                    .build();

            newsVerificationHistoryRepository.save(history);

            return response;
        } catch (WebClientResponseException e) {
            throw new RuntimeException("AI 서버 오류: " + e.getResponseBodyAsString(), e);
        } catch (Exception e) {
            throw new RuntimeException("AI 서버 요청 실패", e);
        }
    }

    public NewsHistoryResponseDTO getHistory(Long userId, NewsHistoryRequestDTO request) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new GeneralException(ErrorInfo.USER_NOT_FOUND));

        Pageable pageable = PageRequest.of(request.getPage(), request.getSize());
        Page<NewsVerificationHistory> historyPage = newsVerificationHistoryRepository.findByUserOrderByIdDesc(user, pageable);

        return NewsHistoryResponseDTO.builder()
                .verifications(historyPage.getContent().stream()
                        .map(history -> NewsVerificationDTO.builder()
                                .id(history.getId())
                                .title(history.getQuery())
                                .build())
                        .toList())
                .pageInfo(NewsHistoryResponseDTO.PageInfo.builder()
                        .currentPage(historyPage.getNumber())
                        .totalPages(historyPage.getTotalPages())
                        .totalElements(historyPage.getTotalElements())
                        .hasNext(historyPage.hasNext())
                        .build())
                .build();
    }

    public NewsHistoryContentResponseDTO getHistoryContent(Long id) {
        NewsVerificationHistory history = newsVerificationHistoryRepository.findById(id)
                .orElseThrow(() -> new GeneralException(ErrorInfo.BAD_REQUEST));

        return NewsHistoryContentResponseDTO.builder()
                .query(history.getQuery())
                .summary(history.getSummary())
                .build();
    }
}

//사용자 ➝ 챗봇 ➝ Spring ➝ Flask ➝ 결과 ➝ Spring ➝ 챗봇 ➝ 사용자