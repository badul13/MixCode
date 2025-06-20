package com.example._42h_tp.controller;

import com.example._42h_tp.dto.request.SignUpRequestDto;
import com.example._42h_tp.dto.response.DataResponseDTO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import com.example._42h_tp.service.UserService;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequestMapping("/user")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    @PostMapping("/signup")
    public ResponseEntity<DataResponseDTO<String>> signUp(@RequestBody SignUpRequestDto signUpRequestDto) {


        log.error(signUpRequestDto.getEmail());
        log.error(signUpRequestDto.getPassword());

        userService.signUp(signUpRequestDto);
        String userEmail = signUpRequestDto.getEmail();
        String message = "회원가입하신 걸 환영합니다, " + userEmail + "님!";

        return ResponseEntity.ok(DataResponseDTO.of(message));
    }
}
