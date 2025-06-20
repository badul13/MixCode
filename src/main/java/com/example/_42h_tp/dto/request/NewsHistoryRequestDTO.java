package com.example._42h_tp.dto.request;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class NewsHistoryRequestDTO {
    private int page = 0;
    private int size = 10;
} 