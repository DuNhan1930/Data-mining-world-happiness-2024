package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

public class Optimizer {
    private double learningRate;
    private double lambda; // L2 Regularization (Weight Decay)

    // Adam parameters
    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;
    private int t = 0;

    // Constructor
    public Optimizer(double learningRate) {
        this(learningRate, 0.0);
    }

    // Constructor có Regularization
    public Optimizer(double learningRate, double lambda) {
        this.learningRate = learningRate;
        this.lambda = lambda;
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
                    // Tính Gradient gốc: Error * Input
                    double grad = n.gradient * layer.inputs[j];

                    // --- APPLY L2 REGULARIZATION ---
                    // Formula: grad_new = grad_old + (lambda * weight_current)
                    grad = grad + (lambda * n.weights[j]);
                    // ---------------------------------

                    // Tính Adam Moments
                    n.mWeights[j] = beta1 * n.mWeights[j] + (1 - beta1) * grad;
                    n.vWeights[j] = beta2 * n.vWeights[j] + (1 - beta2) * (grad * grad);

                    // Bias Correction
                    double mHat = n.mWeights[j] / beta1Correction;
                    double vHat = n.vWeights[j] / beta2Correction;

                    // Update weights
                    n.weights[j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                }

                // 2. Update Bias (Not apply Regularization for Bias)
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