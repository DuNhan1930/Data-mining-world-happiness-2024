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
        // 3. CV configuration
        // =====================
        int seed = 36;
        int numFolds = 10;  // 10-fold cross-validation

        // =========================================================
        // 4. Hyper parameter tuning using 10-fold cross-validation
        // =========================================================

        // --------- 4.1. Tune Random Forest: numIterations (n_estimators) ----------
        int[] rfNumTreesList = {50, 100, 150, 200, 300, 400, 500};

        double bestRfAcc = -1.0;
        int bestRfTrees = rfNumTreesList[0];

        System.out.println("\n=== 10-fold CV: Random Forest hyper parameter search ===");
        for (int numTrees : rfNumTreesList) {
            RandomForest rf = createRandomForest(data, numTrees);

            double cvAcc = crossValidateAccuracy(rf, data, numFolds, seed);
            System.out.printf("Random Forest with %d trees -> CV Accuracy = %.4f%n", numTrees, cvAcc);

            if (cvAcc > bestRfAcc) {
                bestRfAcc = cvAcc;
                bestRfTrees = numTrees;
            }
        }
        System.out.printf("%n[Random Forest] Best numTrees = %d with CV Accuracy = %.4f%n",
                bestRfTrees, bestRfAcc);

        // --------- 4.2. Tune KNN: k neighbors ----------
        int[] knnKList = {1, 3, 5, 7, 9, 11, 13};

        double bestKnnAcc = -1.0;
        int bestK = knnKList[0];

        System.out.println("\n=== 10-fold CV: KNN hyper parameter search ===");
        for (int k : knnKList) {
            IBk knn = createKNN(k);

            double cvAcc = crossValidateAccuracy(knn, data, numFolds, seed);
            System.out.printf("KNN with k=%d -> CV Accuracy = %.4f%n", k, cvAcc);

            // If accuracy is higher, OR
            // if accuracy is (almost) equal and k is larger -> choose this k
            if (cvAcc > bestKnnAcc + 1e-6 ||
                    (Math.abs(cvAcc - bestKnnAcc) <= 1e-6 && k > bestK)) {

                bestKnnAcc = cvAcc;
                bestK = k;
            }
        }

        System.out.printf("%n[KNN] Best k = %d with CV Accuracy = %.4f%n", bestK, bestKnnAcc);

        // =====================
        // 5. Build final models with best params
        // =====================
        RandomForest bestRf = createRandomForest(data, bestRfTrees);
        IBk bestKnn = createKNN(bestK);

        // =====================
        // 6. Final evaluation using 10-fold CV (detailed metrics + per fold)
        // =====================
        evaluateModelCV("Random Forest (best=" + bestRfTrees + " trees)", bestRf, data, numFolds, seed);
        evaluateModelCV("KNN (best k=" + bestK + ")", bestKnn, data, numFolds, seed);
    }

    // -------------------------------------------------
    // Create RandomForest with chosen numTrees
    // -------------------------------------------------
    private static RandomForest createRandomForest(Instances data, int numTrees) {
        RandomForest rf = new RandomForest();

        rf.setNumIterations(numTrees);

        // Number of features per split = sqrt(#attributes - class)
        int numAttributes = data.numAttributes() - 1; // exclude class attribute
        int k = (int) Math.round(Math.sqrt(numAttributes));
        if (k < 1) k = 1;
        rf.setNumFeatures(k);

        rf.setSeed(36);

        // Commented out to reduce noise during loops, uncomment if needed
        // System.out.println("Configured RandomForest: trees=" + rf.getNumIterations() + ", features per split=" + rf.getNumFeatures());
        return rf;
    }

    // -------------------------------------------------
    // Create KNN (IBk) with chosen k
    // -------------------------------------------------
    private static IBk createKNN(int k) {
        IBk knn = new IBk();
        knn.setKNN(k);
        // Commented out to reduce noise during loops, uncomment if needed
        // System.out.println("Configured KNN with k = " + knn.getKNN());
        return knn;
    }

    // -------------------------------------------------
    // Helper: run 10-fold CV and return accuracy only
    // (Used for tuning loop)
    // -------------------------------------------------
    private static double crossValidateAccuracy(Classifier model,
                                                Instances data,
                                                int numFolds,
                                                int seed) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, numFolds, new Random(seed));
        return eval.pctCorrect() / 100.0;
    }

    // -------------------------------------------------
    // Evaluate model with k-fold cross-validation:
    // - PRINTS ACCURACY PER FOLD
    // - Accuracy
    // - Weighted F1
    // - Per-class metrics
    // - Confusion matrix
    // -------------------------------------------------
    private static void evaluateModelCV(String name,
                                        Classifier model,
                                        Instances data,
                                        int numFolds,
                                        int seed) throws Exception {

        System.out.println("\n========================================");
        System.out.println("Model: " + name + " (" + numFolds + "-fold CV)");
        System.out.println("----------------------------------------");
        System.out.println("Per-fold Accuracy Details:");

        // Make a copy of data to randomize
        Instances randData = new Instances(data);
        randData.randomize(new Random(seed));
        if (randData.classAttribute().isNominal()) {
            randData.stratify(numFolds);
        }

        // We need one main evaluation object to aggregate all folds for the final report
        Evaluation totalEval = new Evaluation(randData);

        // Manual CV Loop
        for (int n = 0; n < numFolds; n++) {
            Instances train = randData.trainCV(numFolds, n);
            Instances test = randData.testCV(numFolds, n);

            // We must rebuild the classifier on the training fold
            // (Note: This is what crossValidateModel does internally)
            Classifier foldModel = weka.classifiers.AbstractClassifier.makeCopy(model);
            foldModel.buildClassifier(train);

            // Evaluate on this specific fold
            Evaluation foldEval = new Evaluation(train);
            foldEval.evaluateModel(foldModel, test);

            // Print accuracy for this fold
            double foldAcc = foldEval.pctCorrect();
            System.out.printf("Fold %2d: %.4f%% (%d instances)%n", (n + 1), foldAcc, test.numInstances());

            // Accumulate statistics into the total evaluation object
            totalEval.evaluateModel(foldModel, test);
        }

        // --- Calculate Final Aggregate Metrics ---
        double accuracy   = totalEval.pctCorrect() / 100.0;
        double weightedF1 = totalEval.weightedFMeasure();

        System.out.println("----------------------------------------");
        System.out.printf("Average Accuracy (CV): %.4f%n", accuracy);
        System.out.printf("Weighted F1-score (CV): %.4f%n", weightedF1);

        // ---------- Per-class metrics ----------
        System.out.println("\nPer-class metrics (precision, recall, F1, support):");
        double[][] cm = totalEval.confusionMatrix();

        for (int i = 0; i < data.numClasses(); i++) {
            String className = data.classAttribute().value(i);
            double precision = totalEval.precision(i);
            double recall    = totalEval.recall(i);
            double f1        = totalEval.fMeasure(i);

            int support = 0;
            for (int j = 0; j < data.numClasses(); j++) {
                support += (int) Math.round(cm[i][j]);
            }

            System.out.printf("Class %-10s | precision=%.4f | recall=%.4f | f1=%.4f | support=%d%n",
                    className, precision, recall, f1, support);
        }

        // ---------- Confusion matrix ----------
        System.out.println("\nConfusion matrix (rows=true, cols=pred):");

        // Header row
        System.out.print("          ");
        for (int j = 0; j < data.numClasses(); j++) {
            System.out.printf("%-10s", data.classAttribute().value(j));
        }
        System.out.println();

        // Rows
        for (int i = 0; i < cm.length; i++) {
            System.out.printf("%-10s", data.classAttribute().value(i));
            for (int j = 0; j < cm[i].length; j++) {
                System.out.printf("%-10d", (int) Math.round(cm[i][j]));
            }
            System.out.println();
        }
    }
}