package com.example._42h_tp.dto.response;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Getter;
import lombok.ToString;

@Getter
@ToString
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ResponseDTO {
    private final Boolean success;
    private final String message;

    public ResponseDTO(Boolean success, String message) {
        this.success = success;
        this.message = message;
    }
} 