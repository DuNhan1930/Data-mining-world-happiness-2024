package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

import java.util.List;

public class Optimizer {
    private double learningRate;
    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;
    private int t = 0;

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public void update(Network net) {
        t++;
        double beta1Correction = 1 - Math.pow(beta1, t);
        double beta2Correction = 1 - Math.pow(beta2, t);

        for (Layer layer : net.getLayers()) {
            for (int i = 0; i < layer.neurons.length; i++) {
                Neuron n = layer.neurons[i];

                // 1. Update Weights
                for (int j = 0; j < n.weights.length; j++) {
                    // Gradient = error at neuron * input
                    double grad = n.gradient * layer.inputs[j];

                    n.mWeights[j] = beta1 * n.mWeights[j] + (1 - beta1) * grad;
                    n.vWeights[j] = beta2 * n.vWeights[j] + (1 - beta2) * (grad * grad);

                    double mHat = n.mWeights[j] / beta1Correction;
                    double vHat = n.vWeights[j] / beta2Correction;

                    n.weights[j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                }

                // 2. Update Bias
                double biasGrad = n.gradient;
                n.mBias = beta1 * n.mBias + (1 - beta1) * biasGrad;
                n.vBias = beta2 * n.vBias + (1 - beta2) * (biasGrad * biasGrad);

                double mHatB = n.mBias / beta1Correction;
                double vHatB = n.vBias / beta2Correction;

                n.bias -= learningRate * mHatB / (Math.sqrt(vHatB) + epsilon);
            }
        }
    }
}