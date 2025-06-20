package com.example._42h_tp.security;

import com.example._42h_tp.GeneralException;
import com.example._42h_tp.constant.ErrorInfo;
import com.example._42h_tp.entity.User;
import com.example._42h_tp.repository.UserRepository;
import io.jsonwebtoken.*;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import javax.crypto.SecretKey;
import java.util.Date;

@Slf4j
@Component
@RequiredArgsConstructor
public class JwtUtil {

    private final JwtProperties jwtProperties;

    private final SecretKey secretKey;

    private final UserRepository userRepository;

    public String createAccessToken(Long userId, String role) {
        Date now = new Date();
        Date expiration = new Date(now.getTime() + jwtProperties.getAccessTokenExpiration());

        return Jwts.builder()
                .setSubject("AccessToken")
                .setIssuedAt(now)
                .setExpiration(expiration)
                .claim("userId", userId)
                .claim("role", role)
                .signWith(secretKey, SignatureAlgorithm.HS256)
                .compact();
    }

    @Transactional
    public String createRefreshToken(Long userId) {
        Date now = new Date();
        Date expiration = new Date(now.getTime() + jwtProperties.getRefreshTokenExpiration());

        String refreshToken = Jwts.builder()
                .setSubject("RefreshToken")
                .setIssuedAt(now)
                .setExpiration(expiration)
                .claim("userId", userId)
                .signWith(secretKey, SignatureAlgorithm.HS256)
                .compact();

        User user = userRepository.findById(userId)
                .orElseThrow(() -> new GeneralException(ErrorInfo.USER_NOT_FOUND));
        user.setRefreshToken(refreshToken);
        userRepository.save(user);

        return refreshToken;
    }

    public String resolveToken(HttpServletRequest request) {
        String authorization = request.getHeader("Authorization");

//        if (authorization != null && authorization.startsWith("Bearer ")) {
//            return authorization.substring(7);
//        }

//        return null;
        return authorization;
    }

    public boolean isExpired(String token) {
        try {
            Date expiration = extractAllClaims(token).getExpiration();
            return expiration.before(new Date());
        }
        catch (Exception e) {
            log.warn("JWT 만료 여부 확인 중 예외 발생 : {}", e.getMessage());
            return true;
        }
    }

    public Long getUserId(String token) {
        return extractAllClaims(token).get("userId", Long.class);
    }

    public String getRole(String token) {
        return extractAllClaims(token).get("role", String.class);
    }

    private Claims extractAllClaims(String token) {
        return Jwts.parserBuilder()
                .setSigningKey(secretKey)
                .build()
                .parseClaimsJws(token)
                .getBody();
    }
}
