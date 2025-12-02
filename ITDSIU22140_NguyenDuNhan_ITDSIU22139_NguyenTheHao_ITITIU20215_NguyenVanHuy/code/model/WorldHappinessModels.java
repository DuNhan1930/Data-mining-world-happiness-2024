package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.lazy.IBk;   // KNN

import java.util.Random;

public class WorldHappinessModels {

    // Path to the preprocessed ARFF file.
    private static final String ARFF_PATH =
            "ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy/code/data/World Happiness Report 2024 Preprocessed.arff";

    public static void main(String[] args) throws Exception {

        // =====================
        // 1. Load ARFF data
        // =====================
        DataSource source = new DataSource(ARFF_PATH);
        Instances data = source.getDataSet();

        System.out.println("Loaded data: " + data.numInstances() + " instances, "
                + data.numAttributes() + " attributes.");

        // =====================
        // 2. Set class attribute = Life Ladder Category
        // =====================
        Attribute classAttr = data.attribute("Life Ladder Category");
        if (classAttr == null) {
            // If not found, fallback to last attribute
            System.out.println("WARNING: 'Life Ladder Category' not found, using last attribute as class.");
            data.setClassIndex(data.numAttributes() - 1);
        } else {
            data.setClassIndex(classAttr.index());
        }

        System.out.println("Class attribute: " + data.classAttribute().name());
        System.out.println("Number of classes: " + data.numClasses());

        // =====================
        // 3. Stratified 80/20 train/test split with random seed 42
        // =====================
        int seed = 36;
        double testRatio = 0.2;
        Instances[] split = stratifiedTrainTestSplit(data, testRatio, seed);
        Instances train = split[0];
        Instances test  = split[1];

        System.out.println("Train size: " + train.numInstances());
        System.out.println("Test size : " + test.numInstances());

        // =====================
        // 4. Define models
        // =====================

        // RandomForest
        RandomForest rf = buildRandomForest(train);

        // KNN (IBk)
        IBk knn = buildKNN(train);

        // =====================
        // 5. Train & evaluate models
        // =====================
        evaluateModel("Random Forest", rf, train, test);
        evaluateModel("KNN (k=7)",      knn, train, test);
    }

    // -------------------------------------------------
    // Stratified train/test split
    // -------------------------------------------------
    private static Instances[] stratifiedTrainTestSplit(Instances data,
                                                        double testRatio,
                                                        int seed) {
        Instances rand = new Instances(data);
        rand.randomize(new Random(seed));

        if (!rand.classAttribute().isNominal()) {
            // Fallback simple split if class is not nominal
            int trainSize = (int) Math.round(rand.numInstances() * (1.0 - testRatio));
            int testSize  = rand.numInstances() - trainSize;
            Instances train = new Instances(rand, 0, trainSize);
            Instances test  = new Instances(rand, trainSize, testSize);
            train.setClassIndex(data.classIndex());
            test.setClassIndex(data.classIndex());
            return new Instances[]{ train, test };
        }

        int numClasses = rand.numClasses();
        java.util.List<java.util.List<Integer>> indicesPerClass = new java.util.ArrayList<>();
        for (int c = 0; c < numClasses; c++) {
            indicesPerClass.add(new java.util.ArrayList<>());
        }

        for (int i = 0; i < rand.numInstances(); i++) {
            int cls = (int) rand.instance(i).classValue();
            indicesPerClass.get(cls).add(i);
        }

        Random rnd = new Random(seed);
        Instances train = new Instances(rand, 0);
        Instances test  = new Instances(rand, 0);

        for (int c = 0; c < numClasses; c++) {
            java.util.List<Integer> idxList = indicesPerClass.get(c);
            java.util.Collections.shuffle(idxList, rnd);

            int n = idxList.size();
            int nTrain = (int) Math.round(n * (1.0 - testRatio));
            if (nTrain == n && n > 1) nTrain = n - 1; // ensure at least 1 test if possible

            for (int i = 0; i < n; i++) {
                int origIndex = idxList.get(i);
                if (i < nTrain) {
                    train.add(rand.instance(origIndex));
                } else {
                    test.add(rand.instance(origIndex));
                }
            }
        }

        train.setClassIndex(data.classIndex());
        test.setClassIndex(data.classIndex());

        return new Instances[]{ train, test };
    }

    // -------------------------------------------------
    // Build RandomForest
    // -------------------------------------------------
    private static RandomForest buildRandomForest(Instances train) throws Exception {
        RandomForest rf = new RandomForest();

        // Number of trees
        rf.setNumIterations(300);

        int numAttributes = train.numAttributes() - 1;
        int k = (int) Math.round(Math.sqrt(numAttributes));
        if (k < 1) k = 1;
        rf.setNumFeatures(k);

        rf.setSeed(36);
        rf.buildClassifier(train);

        System.out.println("Built RandomForest with " + rf.getNumIterations() + " trees and "
                + rf.getNumFeatures() + " features per split.");
        return rf;
    }

    // -------------------------------------------------
    // Build KNN (IBk)
    // -------------------------------------------------
    private static IBk buildKNN(Instances train) throws Exception {
        IBk knn = new IBk();

        // k=7 neighbors
        knn.setKNN(7);

        knn.buildClassifier(train);
        System.out.println("Built KNN with k = " + knn.getKNN());
        return knn;
    }

    // -------------------------------------------------
    // Evaluate model: accuracy, weighted F1, per-class metrics, confusion matrix
    // -------------------------------------------------
    private static void evaluateModel(String name,
                                      Classifier model,
                                      Instances train,
                                      Instances test) throws Exception {

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, test);

        double accuracy = eval.pctCorrect() / 100.0;
        double weightedF1 = eval.weightedFMeasure();

        System.out.println("\n========================================");
        System.out.println("Model: " + name);
        System.out.printf("Accuracy: %.4f%n", accuracy);
        System.out.printf("Weighted F1-score: %.4f%n", weightedF1);

        // ---------- Per-class metrics ----------
        System.out.println("\nPer-class metrics (precision, recall, F1, support):");
        for (int i = 0; i < test.numClasses(); i++) {
            String className = test.classAttribute().value(i);
            double precision = eval.precision(i);
            double recall    = eval.recall(i);
            double f1        = eval.fMeasure(i);

            int support = (int) Math.round(eval.numTruePositives(i) + eval.numFalseNegatives(i));

            System.out.printf("Class %-10s | precision=%.4f | recall=%.4f | f1=%.4f | support=%d%n",
                    className, precision, recall, f1, support);
        }

        // ---------- Confusion matrix ----------
        System.out.println("\nConfusion matrix (rows=true, cols=pred):");
        double[][] cm = eval.confusionMatrix();

        // Header row
        System.out.print("          ");
        for (int j = 0; j < test.numClasses(); j++) {
            System.out.printf("%-10s", test.classAttribute().value(j));
        }
        System.out.println();

        // Rows
        for (int i = 0; i < cm.length; i++) {
            System.out.printf("%-10s", test.classAttribute().value(i));
            for (int j = 0; j < cm[i].length; j++) {
                System.out.printf("%-10d", (int) Math.round(cm[i][j]));
            }
            System.out.println();
        }
    }
}
