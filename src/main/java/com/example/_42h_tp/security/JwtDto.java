package com.example._42h_tp.security;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class JwtDto {

    private String accessToken;

    private String refreshToken;

}
