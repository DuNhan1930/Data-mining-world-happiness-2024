package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

import java.util.Random;

public class Neuron {
    public double[] weights;
    public double bias;
    public double output;    // Output after activation
    public double gradient;  // Error gradient (delta)

    // Adam Optimizer Cache
    public double[] mWeights;
    public double[] vWeights;
    public double mBias;
    public double vBias;

    public Neuron(int inputSize, Random rand) {
        weights = new double[inputSize];
        mWeights = new double[inputSize];
        vWeights = new double[inputSize];

        // He Initialization
        double stdDev = Math.sqrt(2.0 / inputSize);
        for (int i = 0; i < inputSize; i++) {
            weights[i] = rand.nextGaussian() * stdDev;
        }
        bias = 0.0;
    }

    // Linear calculation: w * x + b
    public double calculateSum(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sum;
    }

    public void copyWeightsFrom(Neuron source) {
        this.bias = source.bias;
        System.arraycopy(source.weights, 0, this.weights, 0, source.weights.length);
    }
}