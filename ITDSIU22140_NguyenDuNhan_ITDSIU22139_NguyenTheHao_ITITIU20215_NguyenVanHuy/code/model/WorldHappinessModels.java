package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.MultilayerPerceptron;

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
        // 3. Split 80/20 train/test with random seed 42
        //    EN: Randomize, then take first 80% as train, remaining 20% as test.
        //    VI: Trộn ngẫu nhiên (seed=42), lấy 80% đầu để train, 20% còn lại để test.
        // =====================
        int seed = 42;
        Instances randData = new Instances(data);
        randData.randomize(new Random(seed));
        if (randData.classAttribute().isNominal()) {
            // EN: This keeps class distribution more balanced across the split.
            // VI: Stratify giúp giữ tỉ lệ class gần giống nhau giữa train và test.
            randData.stratify(5); // optional: 5 folds for better class balance
        }

        int trainSize = (int) Math.round(randData.numInstances() * 0.8);
        int testSize  = randData.numInstances() - trainSize;

        Instances train = new Instances(randData, 0, trainSize);
        Instances test  = new Instances(randData, trainSize, testSize);

        System.out.println("Train size: " + train.numInstances());
        System.out.println("Test size : " + test.numInstances());

        // =====================
        // 4. Define models
        // =====================

        // RandomForest ≈ sklearn RandomForestClassifier
        RandomForest rf = buildRandomForest(train);

        // MultilayerPerceptron ≈ sklearn MLPClassifier
        MultilayerPerceptron mlp = buildMLP(train);

        // =====================
        // 5. Train & evaluate models
        // =====================
        evaluateModel("Random Forest", rf, train, test);
        evaluateModel("MLP",            mlp, train, test);
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
        // VI: Số epoch/bước huấn luyện, giống max_iter.
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
