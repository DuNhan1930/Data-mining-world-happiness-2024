package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

public class ActivationFunction {

    // Rectified Linear Unit: f(x) = max(0, x)
    public static double relu(double x) {
        return Math.max(0, x);
    }

    // Derivative of ReLU: f'(x) = 1 if x > 0 else 0
    public static double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    // Softmax: turns vector into probabilities summing to 1
    public static double[] softmax(double[] z) {
        double[] out = new double[z.length];

        // Find max for numerical stability (prevents overflow)
        double max = z[0];
        for (double v : z) if (v > max) max = v;

        double sum = 0;
        for (int i = 0; i < z.length; i++) {
            out[i] = Math.exp(z[i] - max);
            sum += out[i];
        }

        // Normalize
        for (int i = 0; i < z.length; i++) {
            out[i] /= sum;
        }
        return out;
    }
}