package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

import java.util.ArrayList;
import java.util.List;

public class Network {
    private List<Layer> layers = new ArrayList<>();

    // Dropout rate: 20% of neurons will be turned off during training
    private double dropoutRate = 0.2;

    public Network(int inputSize) {
        // --- Architecture for High-Dimensional Data (~173 inputs) ---
        // Input -> Hidden 1 (128 neurons) -> Hidden 2 (64 neurons) -> Output (3 classes)
        layers.add(new Layer(inputSize, 128));
        layers.add(new Layer(128, 64));
        layers.add(new Layer(64, 3));
    }

    /**
     * Forward Pass with Inverted Dropout
     * @param input The input vector
     * @param isTraining true if training (apply dropout), false if testing (no dropout)
     * @return The output probabilities
     */
    public double[] forward(double[] input, boolean isTraining) {
        double[] current = input;

        for (int i = 0; i < layers.size(); i++) {
            Layer l = layers.get(i);
            l.inputs = current;
            double[] z = new double[l.neurons.length];

            // 1. Calculate Linear Sum (z = Wx + b)
            for (int j = 0; j < l.neurons.length; j++) {
                z[j] = l.neurons[j].calculateSum(current);
            }

            // 2. Activation & Dropout Logic
            if (i == layers.size() - 1) {
                // Output Layer: Always Softmax, No Dropout
                l.outputs = ActivationFunction.softmax(z);
                // Fill mask with 1.0s (keep all) for safety
                for(int j=0; j<l.neurons.length; j++) l.dropoutMask[j] = 1.0;
            } else {
                // Hidden Layers: ReLU + Dropout
                l.outputs = new double[z.length];
                // Inverted Dropout Scaling Factor
                double scale = 1.0 / (1.0 - dropoutRate);

                for (int j = 0; j < z.length; j++) {
                    l.outputs[j] = ActivationFunction.relu(z[j]);

                    if (isTraining) {
                        // Randomly drop neurons
                        if (Math.random() < dropoutRate) {
                            l.dropoutMask[j] = 0.0; // Drop
                            l.outputs[j] = 0.0;
                        } else {
                            l.dropoutMask[j] = scale; // Keep & Scale
                            l.outputs[j] *= scale;
                        }
                    } else {
                        l.dropoutMask[j] = 1.0;
                    }
                }
            }

            // Sync calculation results back to Neuron objects
            for(int j=0; j<l.neurons.length; j++) {
                l.neurons[j].output = l.outputs[j];
            }

            current = l.outputs;
        }
        return current;
    }

    public void backward(double[] target) {
        // 1. Calculate Output Layer Gradients
        Layer outputLayer = layers.get(layers.size() - 1);
        double[] outputGrads = LossFunction.getGradient(outputLayer.outputs, target);

        for(int i=0; i<outputLayer.neurons.length; i++) {
            outputLayer.neurons[i].gradient = outputGrads[i];
        }

        // 2. Backpropagate through Hidden Layers
        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer current = layers.get(i);
            Layer next = layers.get(i + 1);

            for (int j = 0; j < current.neurons.length; j++) {
                double errorSum = 0;
                // Sum weighted errors from the next layer
                for (int k = 0; k < next.neurons.length; k++) {
                    // Weight connecting current[j] to next[k]
                    errorSum += next.neurons[k].weights[j] * next.neurons[k].gradient;
                }

                // Calculate gradient: Error * Derivative * DropoutMask
                // If the neuron was dropped (mask=0), the gradient becomes 0
                double derivative = ActivationFunction.reluDerivative(current.neurons[j].output);
                current.neurons[j].gradient = errorSum * derivative * current.dropoutMask[j];
            }
        }
    }

    // Method to Deep Copy weights (Required for "Restore Best Weights")
    public void copyWeightsFrom(Network source) {
        for (int i = 0; i < this.layers.size(); i++) {
            this.layers.get(i).copyWeightsFrom(source.layers.get(i));
        }
    }

    public List<Layer> getLayers() {
        return layers;
    }
}