package com.example._42h_tp.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientConfig {

    @Bean
    public WebClient aiServerWebClient() {
        return WebClient.builder()
                .baseUrl("https://9ffe-34-142-192-104.ngrok-free.app")
                .build();
    }
}