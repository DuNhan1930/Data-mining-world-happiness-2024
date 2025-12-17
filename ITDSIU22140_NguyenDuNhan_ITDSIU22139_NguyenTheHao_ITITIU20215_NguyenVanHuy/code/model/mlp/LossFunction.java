package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

public class LossFunction {

    // Calculate Loss: -sum(target * log(prediction))
    public static double compute(double[] prediction, double[] target) {
        double loss = 0;
        for (int i = 0; i < prediction.length; i++) {
            // Clip value to avoid log(0)
            double val = Math.max(prediction[i], 1e-15);
            loss -= target[i] * Math.log(val);
        }
        return loss;
    }

    // Gradient of Loss w.r.t Softmax input (z)
    // Result is simply: Prediction - Target
    public static double[] getGradient(double[] prediction, double[] target) {
        double[] grad = new double[prediction.length];
        for (int i = 0; i < prediction.length; i++) {
            grad[i] = prediction[i] - target[i];
        }
        return grad;
    }
}