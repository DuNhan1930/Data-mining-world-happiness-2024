package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;   // KNN

import java.util.Random;

public class WorldHappinessModels {

    // EN: Path to the preprocessed ARFF file.
    // VI: Đường dẫn tới file ARFF đã tiền xử lý.
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
        //    EN: Similar idea to train_test_split(..., stratify=y, test_size=0.2, random_state=42)
        // =====================
        int seed = 42;
        double testRatio = 0.2;
        Instances[] split = stratifiedTrainTestSplit(data, testRatio, seed);
        Instances train = split[0];
        Instances test  = split[1];

        System.out.println("Train size: " + train.numInstances());
        System.out.println("Test size : " + test.numInstances());

        // =====================
        // 4. Define models
        // =====================

        // RandomForest ≈ sklearn RandomForestClassifier
        RandomForest rf = buildRandomForest(train);

        // MultilayerPerceptron ≈ sklearn MLPClassifier (but different backend)
        MultilayerPerceptron mlp = buildMLP(train);

        // KNN (IBk) ≈ KNeighborsClassifier(n_neighbors=5)
        IBk knn = buildKNN(train);

        // =====================
        // 5. Train & evaluate models
        // =====================
        evaluateModel("Random Forest", rf, train, test);
        evaluateModel("MLP",            mlp, train, test);
        evaluateModel("KNN (k=5)",      knn, train, test);
    }

    // -------------------------------------------------
    // Stratified train/test split (like train_test_split with stratify=y)
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

        // Manual stratification per class (giống stratify=y trong Python)
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
    // Build RandomForest (like RandomForestClassifier)
    // -------------------------------------------------
    private static RandomForest buildRandomForest(Instances train) throws Exception {
        RandomForest rf = new RandomForest();

        // EN: Number of trees (similar to n_estimators in sklearn).
        // VI: Số cây trong rừng (gần giống n_estimators).
        rf.setNumIterations(300); // newer Weka uses setNumIterations instead of setNumTrees

        // EN: max_features='sqrt' → use sqrt(numAttributes) features per split.
        // VI: max_features='sqrt' → mỗi node chọn sqrt(M) thuộc tính ngẫu nhiên.
        int numAttributes = train.numAttributes() - 1; // exclude class
        int k = (int) Math.round(Math.sqrt(numAttributes));
        if (k < 1) k = 1;
        rf.setNumFeatures(k);

        rf.setSeed(42);
        rf.buildClassifier(train);

        System.out.println("Built RandomForest with " + rf.getNumIterations() + " trees and "
                + rf.getNumFeatures() + " features per split.");
        return rf;
    }

    // -------------------------------------------------
    // Build MLP (like MLPClassifier(hidden_layer_sizes=(64,32)))
    // -------------------------------------------------
    private static MultilayerPerceptron buildMLP(Instances train) throws Exception {
        MultilayerPerceptron mlp = new MultilayerPerceptron();

        // EN: Hidden layers "64,32" ≈ (64, 32) in sklearn.
        // VI: 2 hidden layer, lần lượt 64 và 32 neurons.
        mlp.setHiddenLayers("64,32");

        // EN: TrainingTime ≈ max_iter.
        // VI: Số epoch/bước huấn luyện, gần giống max_iter.
        mlp.setTrainingTime(500);

        // You can tune these if needed
        mlp.setLearningRate(0.3);
        mlp.setMomentum(0.2);
        mlp.setSeed(42);

        mlp.buildClassifier(train);
        System.out.println("Built MLP with hidden layers: " + mlp.getHiddenLayers());
        return mlp;
    }

    // -------------------------------------------------
    // Build KNN (IBk) ≈ KNeighborsClassifier(n_neighbors=5)
    // -------------------------------------------------
    private static IBk buildKNN(Instances train) throws Exception {
        IBk knn = new IBk();

        // k=5 neighbors
        knn.setKNN(5);

        // No weighting (classic KNN)
        // Weka 3.x uses this automatically → equal weighting

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

            // EN: support = TP + FN (true instances of that class in test set)
            // VI: support = số mẫu thực sự thuộc lớp này trong tập test.
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
