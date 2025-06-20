package com.example._42h_tp.dto.response;

import com.example._42h_tp.constant.ErrorInfo;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ErrorResponseDTO extends ResponseDTO {
    private final String error;
    private final String timestamp;
    private final int status;

    private ErrorResponseDTO(ErrorInfo errorInfo, String message) {
        super(false, message);
        this.error = errorInfo.name();
        this.timestamp = LocalDateTime.now().toString();
        this.status = errorInfo.getHttpStatus().value();
    }

    public static ErrorResponseDTO of(ErrorInfo errorInfo, String message) {
        return new ErrorResponseDTO(errorInfo, message);
    }
} 