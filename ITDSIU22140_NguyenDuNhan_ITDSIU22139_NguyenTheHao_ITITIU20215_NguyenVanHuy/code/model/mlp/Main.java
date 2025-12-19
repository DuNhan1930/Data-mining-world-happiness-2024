package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;

public class Main {
    private static final String ARFF_PATH =
            "ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy/code/data/World Happiness Report 2024 Preprocessed.arff";

    // --- Hyperparameters ---
    private static final int MAX_EPOCHS = 200;
    private static final double LEARNING_RATE = 0.0001;
    private static final int FOLDS = 10;

    // --- Early Stopping Parameters ---
    private static final int PATIENCE = 20;
    private static final int MIN_EPOCHS_BEFORE_STOP = 40;
    private static final double MIN_DELTA = 1e-4;

    public static void main(String[] args) throws Exception {
        // 1. Load Data
        Instances data = WekaUtils.loadData(ARFF_PATH);

        // 2. Balance Data
        // System.out.println("Applying Class Balancing...");
        // data = WekaUtils.balanceData(data);

        // 3. Normalization
        System.out.println("Applying Normalization...");
        data = WekaUtils.normalizeData(data);
        // --------------------------------------

        data.randomize(new Random(42));

        int inputSize = data.numAttributes() - 1;
        System.out.println("Detected Input Size: " + inputSize);

        double totalAccuracy = 0;
        int[][] globalConfusionMatrix = new int[3][3];

        System.out.println("Starting MLP with Early Stopping & Weight Restoration...");

        for (int n = 0; n < FOLDS; n++) {
            System.out.println("\n=== Fold " + (n + 1) + " ===");

            Instances trainOriginal = data.trainCV(FOLDS, n);
            Instances testSet = data.testCV(FOLDS, n);

            trainOriginal.randomize(new Random(n));
            int valSize = (int) (trainOriginal.numInstances() * 0.10);
            int trainRealSize = trainOriginal.numInstances() - valSize;

            Instances trainActual = new Instances(trainOriginal, 0, trainRealSize);
            Instances valSet = new Instances(trainOriginal, trainRealSize, valSize);

            Network model = new Network(inputSize);
            Network bestModel = new Network(inputSize);
            Optimizer optimizer = new Optimizer(LEARNING_RATE, 0.001);

            double best_val_loss = Double.MAX_VALUE;
            int best_epoch = 0;
            int wait = 0;

            for (int epoch = 1; epoch <= MAX_EPOCHS; epoch++) {
                // Training
                trainActual.randomize(new Random(epoch));
                for (int i = 0; i < trainActual.numInstances(); i++) {
                    Instance inst = trainActual.instance(i);
                    model.forward(WekaUtils.instanceToInput(inst), true);
                    model.backward(WekaUtils.targetToOneHot(inst));
                    optimizer.update(model);
                }

                // Validation
                double val_loss = calculateLoss(model, valSet);

                // Check improvement
                if (best_val_loss - val_loss > MIN_DELTA) {
                    best_val_loss = val_loss;
                    best_epoch = epoch;
                    bestModel.copyWeightsFrom(model);
                    wait = 0;

                    // Just print log when have new record
                    if (epoch % 5 == 0) {
                        System.out.printf("   [+] Epoch %d: New Best Loss %.5f\n", epoch, val_loss);
                    }
                } else {
                    wait++;
                }

                // Early Stopping Condition
                if (epoch > MIN_EPOCHS_BEFORE_STOP && wait >= PATIENCE) {
                    System.out.println("   [STOP] Early Stopping triggered at Epoch " + epoch);
                    break;
                }
            }

            System.out.println("   [RESTORE] Loading weights from Best Epoch: " + best_epoch);
            if (best_epoch > 0) {
                model.copyWeightsFrom(bestModel);
            }

            updateConfusionMatrix(model, testSet, globalConfusionMatrix);
            double foldAccuracy = evaluate(model, testSet);
            totalAccuracy += foldAccuracy;
            System.out.printf("-> Fold %d Test Accuracy: %.2f%%\n", (n + 1), foldAccuracy * 100);
        }

        System.out.println("\n-------------------------------------");
        System.out.printf("Final Average Test Accuracy: %.2f%%\n", (totalAccuracy / FOLDS) * 100);
        printConfusionMatrix(globalConfusionMatrix, data);
    }

    private static double calculateLoss(Network net, Instances data) {
        double totalLoss = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            double[] output = net.forward(WekaUtils.instanceToInput(inst), false);
            totalLoss += LossFunction.compute(output, WekaUtils.targetToOneHot(inst));
        }
        return totalLoss / data.numInstances();
    }

    private static double evaluate(Network net, Instances data) {
        int correct = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            double[] output = net.forward(WekaUtils.instanceToInput(inst), false);
            if (getArgMax(output) == (int) inst.classValue()) correct++;
        }
        return (double) correct / data.numInstances();
    }

    private static void updateConfusionMatrix(Network net, Instances data, int[][] matrix) {
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            double[] output = net.forward(WekaUtils.instanceToInput(inst), false);
            matrix[(int) inst.classValue()][getArgMax(output)]++;
        }
    }

    private static int getArgMax(double[] arr) {
        int maxIdx = 0;
        double maxVal = arr[0];
        for (int i = 1; i < arr.length; i++) if (arr[i] > maxVal) { maxVal = arr[i]; maxIdx = i; }
        return maxIdx;
    }

    private static void printConfusionMatrix(int[][] matrix, Instances data) {
        System.out.println("\n=== Overall Confusion Matrix ===");
        int numClasses = data.numClasses();
        String[] classNames = new String[numClasses];
        for(int i=0; i<numClasses; i++) classNames[i] = data.classAttribute().value(i);
        System.out.print("\t");
        for (String name : classNames) System.out.printf("%-10s ", name + "(P)");
        System.out.println();
        for (int i = 0; i < numClasses; i++) {
            System.out.printf("%-10s ", classNames[i] + "(A)");
            for (int j = 0; j < numClasses; j++) System.out.printf("%-10d ", matrix[i][j]);
            System.out.println();
        }
    }
}