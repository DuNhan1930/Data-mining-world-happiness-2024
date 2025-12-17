package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;
import java.util.Arrays;

public class Main {
    private static final String ARFF_PATH =
            "ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy/code/data/World Happiness Report 2024 Preprocessed.arff";

    private static final int EPOCHS = 50;
    private static final double LEARNING_RATE = 0.001;
    private static final int FOLDS = 10;
    private static final int NUM_CLASSES = 3; // Low, Medium, High

    public static void main(String[] args) throws Exception {
        // 1. Load Data
        Instances data = WekaUtils.loadData(ARFF_PATH);
        data.randomize(new Random(42));

        // Auto-detect input size (columns - class column)
        int inputSize = data.numAttributes() - 1;
        System.out.println("Detected Input Size: " + inputSize);

        // Matrix to store results across all 10 folds
        int[][] confusionMatrix = new int[NUM_CLASSES][NUM_CLASSES];
        double totalAccuracy = 0;

        System.out.println("Starting MLP Training (10-Fold CV)...");
        System.out.println("-------------------------------------");

        for (int n = 0; n < FOLDS; n++) {
            Instances train = data.trainCV(FOLDS, n);
            Instances test = data.testCV(FOLDS, n);

            Network net = new Network(inputSize);
            Optimizer optimizer = new Optimizer(LEARNING_RATE);

            // --- Training Loop ---
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                train.randomize(new Random(epoch)); // Shuffle for better stability

                for (int i = 0; i < train.numInstances(); i++) {
                    Instance inst = train.instance(i);
                    double[] input = WekaUtils.instanceToInput(inst);
                    double[] target = WekaUtils.targetToOneHot(inst);

                    net.forward(input);
                    net.backward(target);
                    optimizer.update(net);
                }
            }

            // --- Evaluation Loop ---
            int correct = 0;
            for (int i = 0; i < test.numInstances(); i++) {
                Instance inst = test.instance(i);
                double[] input = WekaUtils.instanceToInput(inst);
                double[] output = net.forward(input);

                int actual = (int) inst.classValue();
                int predicted = getArgMax(output);

                // Update Matrix: Row = Actual, Col = Predicted
                confusionMatrix[actual][predicted]++;

                if (predicted == actual) {
                    correct++;
                }
            }

            double foldAccuracy = (double) correct / test.numInstances();
            totalAccuracy += foldAccuracy;
            System.out.printf("Fold %d: Accuracy = %.2f%%\n", (n + 1), foldAccuracy * 100);
        }

        System.out.println("-------------------------------------");
        System.out.printf("Final Average Accuracy: %.2f%%\n", (totalAccuracy / FOLDS) * 100);
        System.out.println("-------------------------------------");

        // Print Confusion Matrix
        printConfusionMatrix(confusionMatrix, data);
    }

    private static int getArgMax(double[] arr) {
        int maxIdx = 0;
        double maxVal = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private static void printConfusionMatrix(int[][] matrix, Instances data) {
        System.out.println("=== Overall Confusion Matrix ===");

        // Get class names from Weka metadata
        int numClasses = data.numClasses();
        String[] classNames = new String[numClasses];
        for(int i=0; i<numClasses; i++) {
            classNames[i] = data.classAttribute().value(i);
        }

        // Print Header
        System.out.print("\t");
        for (String name : classNames) {
            System.out.printf("%-10s ", name + "(P)");
        }
        System.out.println();

        // Print Rows
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%-10s ", classNames[i] + "(A)"); // A for Actual
            for (int j = 0; j < numClasses; j++) {
                System.out.printf("%-10d ", matrix[i][j]);
            }
            System.out.println();
        }

        System.out.println("\n(A) = Actual, (P) = Predicted");
    }
}