package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

import java.util.Random;

public class Layer {
    public Neuron[] neurons;
    public double[] inputs;
    public double[] outputs;

    // New: Store the dropout mask for backprop
    // Contains 0.0 (dropped) or 1/(1-rate) (kept)
    public double[] dropoutMask;

    public Layer(int numInputs, int numNeurons) {
        neurons = new Neuron[numNeurons];
        outputs = new double[numNeurons];
        dropoutMask = new double[numNeurons]; // Init mask array

        Random rand = new Random();
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(numInputs, rand);
        }
    }

    public void copyWeightsFrom(Layer source) {
        for (int i = 0; i < this.neurons.length; i++) {
            this.neurons[i].copyWeightsFrom(source.neurons[i]);
        }
    }
}