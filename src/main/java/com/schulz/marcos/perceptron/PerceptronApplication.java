package com.schulz.marcos.perceptron;

import lombok.extern.log4j.Log4j2;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@Log4j2
public class PerceptronApplication {

    /**
     * main method.
     * apebas inicia o perceptron, solicita o treino e apos processos, imprime os pesos finais.
     * @param args
     */
    public static void main(String[] args) {
        SpringApplication.run(PerceptronApplication.class, args);
        Perceptron perceptron = new Perceptron();
        perceptron.train();
        double[] weights = perceptron.getWeights();
        log.info(
                "Weights corrects: w1:{}, w2:{},w3:{}, w4:{}, w5:{}, w6:{}, w7:{}",
                weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6]
        );

    }
}
