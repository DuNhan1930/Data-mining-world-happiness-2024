package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

import java.util.Random;

public class Layer {
    public Neuron[] neurons;
    public double[] inputs;  // Cache inputs for backprop
    public double[] outputs; // Cache outputs for next layer

    public Layer(int numInputs, int numNeurons) {
        neurons = new Neuron[numNeurons];
        outputs = new double[numNeurons];
        Random rand = new Random();

        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(numInputs, rand);
        }
    }
}