package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

import java.util.ArrayList;
import java.util.List;

public class Network {
    private List<Layer> layers = new ArrayList<>();

    public Network(int inputSize) {
        layers.add(new Layer(inputSize, 128));
        layers.add(new Layer(128, 64));
        layers.add(new Layer(64, 3));
    }

    public double[] forward(double[] input) {
        double[] current = input;

        for (int i = 0; i < layers.size(); i++) {
            Layer l = layers.get(i);
            l.inputs = current;
            double[] z = new double[l.neurons.length];

            // Calculate Sums
            for (int j = 0; j < l.neurons.length; j++) {
                z[j] = l.neurons[j].calculateSum(current);
            }

            // Activation
            if (i == layers.size() - 1) {
                l.outputs = ActivationFunction.softmax(z);
            } else {
                l.outputs = new double[z.length];
                for (int j = 0; j < z.length; j++) {
                    l.outputs[j] = ActivationFunction.relu(z[j]);
                }
            }

            // Update neuron output state
            for(int j=0; j<l.neurons.length; j++) {
                l.neurons[j].output = l.outputs[j];
            }

            current = l.outputs;
        }
        return current;
    }

    public void backward(double[] target) {
        // 1. Output Layer
        Layer outputLayer = layers.get(layers.size() - 1);
        double[] outputGrads = LossFunction.getGradient(outputLayer.outputs, target);
        for(int i=0; i<outputLayer.neurons.length; i++) {
            outputLayer.neurons[i].gradient = outputGrads[i];
        }

        // 2. Hidden Layers
        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer current = layers.get(i);
            Layer next = layers.get(i + 1);

            for (int j = 0; j < current.neurons.length; j++) {
                double errorSum = 0;
                for (int k = 0; k < next.neurons.length; k++) {
                    // Access weight k->j (Weight of Neuron K at index J)
                    errorSum += next.neurons[k].weights[j] * next.neurons[k].gradient;
                }
                current.neurons[j].gradient = errorSum * ActivationFunction.reluDerivative(current.neurons[j].output);
            }
        }
    }

    public List<Layer> getLayers() { return layers; }
}