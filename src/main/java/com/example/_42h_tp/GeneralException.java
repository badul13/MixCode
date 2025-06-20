package com.example._42h_tp;

import com.example._42h_tp.constant.ErrorInfo;
import lombok.Getter;

@Getter
public class GeneralException extends RuntimeException {

    private final ErrorInfo errorInfo;

    public GeneralException(ErrorInfo errorInfo) {
        super(errorInfo.getMessage());
        this.errorInfo = errorInfo;
    }
}