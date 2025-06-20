package com.example._42h_tp.dto.response;

import lombok.Getter;

@Getter
public class DataResponseDTO<T> extends ResponseDTO {

    private final T data;

    private DataResponseDTO(T data, String message) {
        super(true, message);
        this.data = data;
    }

    private DataResponseDTO(String message) {
        super(true, message);
        this.data = null;
    }

    public static <T> DataResponseDTO<T> of(T data, String message) {
        return new DataResponseDTO<>(data, message);
    }

    public static <T> DataResponseDTO<T> of(String message) {
        return new DataResponseDTO<>(message);
    }
} 