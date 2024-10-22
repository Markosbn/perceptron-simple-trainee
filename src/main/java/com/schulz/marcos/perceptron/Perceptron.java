package com.schulz.marcos.perceptron;

import lombok.Getter;
import lombok.extern.log4j.Log4j2;

import java.util.HashMap;
import java.util.Map;

@Log4j2
public class Perceptron {
    /**
     * Código aluno / perceptron base de conhecimento.
     */
    private static final String RU_SAMPLE_BASE = "4138882";
    private static final int SIZE_PERCEPTRON = RU_SAMPLE_BASE.length();

    @Getter
    private double[] weights;
    private boolean trained;
    private Map<Integer, Double> epochRate;

    private SampleDto[] samples;
    private double bias;
    private double learningRate;

    public Perceptron() {
        this(0.1, 0);
    }

    /**
     * construtor que inicializa amostra e pesos com valores fixos. Juntamente com informações pertinentes para o controle do perceptron.
     * @param learningRate - também conhecido como (k)
     * @param bias - também conhhecido como (w0)
     */
    public Perceptron(double learningRate, double bias) {
        this.epochRate = new HashMap<>();
        this.trained = false;
        this.weights = new double[SIZE_PERCEPTRON];
        this.learningRate = learningRate; //(k)
        this.bias = bias;

        //Gerar amostrar com dados fixos
        this.samples = new SampleDto[]{
                new SampleDto("4128882", -1),
                new SampleDto("4138891", 1),
                new SampleDto("4148882", 1),
                new SampleDto("4138671", -1),
                new SampleDto(RU_SAMPLE_BASE, 1)
        };

        for (int i = 0; i < SIZE_PERCEPTRON; i++) {
            weights[i] = 1;
        }
    }

    /**
     * train() é o metodo que realiza o treino do perceptron.
     * Pega o conjunto de amostrar e processa uma epoca, até que uma epoca feche pelo menos 80% de acerto.
     */
    public void train() {
        int epochControl = 0;
        while (!this.trained) {
            epochControl++;
            int hits = 0;
            for (SampleDto sample : this.samples) {
                double net = calculateNet(sample.getSampleRu());
                int responsePercepton = activateFunction(net);

                if (responsePercepton != sample.getExpectedResult()) {
                    int error = calculateError(sample, responsePercepton);
                    double[] newWeights = calculateNewWeights(sample, error);
                    this.weights = newWeights;
                } else {
                    hits ++;
                }
            }
            double percentageHits = calculatePercentageHits(hits);
            this.epochRate.put(epochControl, percentageHits);

            log.info("Epoch {}. Hits rate: {}%. Hits: {}. Weights: ", epochControl, percentageHits, hits);
            for (int j = 0; j < SIZE_PERCEPTRON; j++) {
               log.info("Wheight {}: {}", j + 1, weights[j]);
            }

            if (percentageHits > 79.99) this.trained = true;
        }
    }

    /**
     * recebe quantidade de acertos e devolve o percentual de acertos.
     * @param hits
     * @return
     */
    private double calculatePercentageHits(int hits) {
        return (double) (hits * 100) / this.samples.length;
    }

    /**
     * recebe amostra e a resposta do perceptron e devolve o valor de erro (e)
     * @param sample
     * @param responsePercepton
     * @return
     */
    private int calculateError(SampleDto sample, int responsePercepton) {
        return sample.getExpectedResult() - responsePercepton;
    }

    /**
     * recebe a amostra e o valore de erro para calcular o novo peso usando a formula delta
     * @param sample
     * @param error - (e)
     * @return
     */
    private double[] calculateNewWeights(SampleDto sample, int error) {
        double[] newWeights = new double[SIZE_PERCEPTRON];
        for (int i = 0; i < SIZE_PERCEPTRON; i++) {
            Integer[] ruArray = getRuArray(sample.getSampleRu());
             double deltaWeigght = learningRate * error * ruArray[i];
            newWeights[i] = this.weights[i] + deltaWeigght;
        }
        return newWeights;
    }

    /**
     * recebe o ru de amostra e calcula o net usando os pesos.
     * @param sampleRu
     * @return
     */
    private double calculateNet(String sampleRu) {
        double net = this.bias;
        for (int i = 0; i < SIZE_PERCEPTRON; i++) {
            Integer[] ruArray = getRuArray(sampleRu);
            net = net + (ruArray[i] * this.weights[i]);
        }
        return net;
    }

    /**
     * funçao de ativaçao, recebe o valor de net e devolve a funçao aplicada, com base na regra desejada.
     * @param net
     * @return
     */
    private int activateFunction(double net){
        return net >= 0d ? 1 : -1;
    }

    /**
     * recebe uma string de amostra de ru e devolve um array de inteiro.
     * @param ruSample
     * @return
     */
    private static Integer[] getRuArray(String ruSample) {
        return ruSample.chars()
                .mapToObj(c -> Integer.parseInt(String.valueOf((char) c)))
                .toArray(Integer[]::new);
    }

    /**
     * recepe um ru e retorna o valor da previsão, 1 ou -1.
     * @param sampleRu
     * @return
     */
    public int predict(String sampleRu) {
        if (!this.trained) return 0;
        double net = this.calculateNet(sampleRu);
        return this.activateFunction(net);
    }
}
