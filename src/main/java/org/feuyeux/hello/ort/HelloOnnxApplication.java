package org.feuyeux.hello.ort;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@Slf4j
public class HelloOnnxApplication {

	public static void main(String[] args) {
		SpringApplication.run(HelloOnnxApplication.class, args);
	}

}
