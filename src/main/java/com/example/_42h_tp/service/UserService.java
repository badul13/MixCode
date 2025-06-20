package com.example._42h_tp.service;

import com.example._42h_tp.GeneralException;
import com.example._42h_tp.constant.ErrorInfo;
import com.example._42h_tp.dto.request.SignUpRequestDto;
import com.example._42h_tp.entity.User;
import com.example._42h_tp.repository.UserRepository;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Slf4j
@Service
@RequiredArgsConstructor
public class UserService {

    private final UserRepository userRepository;

    private final PasswordEncoder passwordEncoder;

    @Transactional
    public void signUp(SignUpRequestDto signUpRequestDto) {

        if (userRepository.existsByEmail(signUpRequestDto.getEmail())) {
            throw new GeneralException(ErrorInfo.EMAIL_ALREADY_EXIST);
        }

        User signUpUser = User.builder()
                .role("ROLE_USER")
                .email(signUpRequestDto.getEmail())
                .password(passwordEncoder.encode(signUpRequestDto.getPassword()))
                .build();

        userRepository.save(signUpUser);
    }
}
