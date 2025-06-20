package com.example._42h_tp.controller;

import com.example._42h_tp.dto.request.NewsHistoryRequestDTO;
import com.example._42h_tp.dto.request.NewsVerifyRequestDTO;
import com.example._42h_tp.dto.response.DataResponseDTO;
import com.example._42h_tp.dto.response.NewsHistoryResponseDTO;
import com.example._42h_tp.dto.response.NewsVerifyResponseDTO;
import com.example._42h_tp.dto.response.NewsHistoryContentResponseDTO;
import com.example._42h_tp.service.NewsService;
import com.example._42h_tp.security.JwtUtil;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/news")
@RequiredArgsConstructor
public class NewsController {

    private final NewsService newsService;
    private final JwtUtil jwtUtil;

    @PostMapping("/chat")
    public ResponseEntity<DataResponseDTO<NewsVerifyResponseDTO>> verifyNews(
            HttpServletRequest request,
            @RequestBody NewsVerifyRequestDTO newsVerifyRequestDTO) {
        
        String token = jwtUtil.resolveToken(request);
        Long userId = jwtUtil.getUserId(token);
        
        NewsVerifyResponseDTO response = newsService.verify(userId, newsVerifyRequestDTO);
//        NewsVerifyResponseDTO response = newsService.verify(1L, newsVerifyRequestDTO);
        return ResponseEntity.ok(DataResponseDTO.of(response, "뉴스 검증 완료"));
    }

    @GetMapping("/history")
    public ResponseEntity<DataResponseDTO<NewsHistoryResponseDTO>> getHistory(
            HttpServletRequest request,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        
        String token = jwtUtil.resolveToken(request);
        Long userId = jwtUtil.getUserId(token);
        
        NewsHistoryRequestDTO requestDTO = new NewsHistoryRequestDTO();
        requestDTO.setPage(page);
        requestDTO.setSize(size);
        
        NewsHistoryResponseDTO response = newsService.getHistory(userId, requestDTO);
        return ResponseEntity.ok(DataResponseDTO.of(response, "검증 기록 조회 성공"));
    }

    @GetMapping("/historyContent/{id}")
    public ResponseEntity<DataResponseDTO<NewsHistoryContentResponseDTO>> getHistoryContent(
            @PathVariable Long id) {
        NewsHistoryContentResponseDTO response = newsService.getHistoryContent(id);
        return ResponseEntity.ok(DataResponseDTO.of(response, "기록 내용 조회 성공"));
    }
}
