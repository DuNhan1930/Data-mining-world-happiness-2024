package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.modeling;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;import weka.classifiers.lazy.IBk;

import weka.core.Instances;
import weka.core.Attribute;

public class ClassifierTrainer {

    public static class TrainResult {
        public final Classifier model;
        public final Instances trainingData;
        public TrainResult(Classifier model, Instances trainingData) {
            this.model = model; this.trainingData = trainingData;
        }
    }

    public static TrainResult trainRandomForest(Instances data, int classIndex) throws Exception {
        // Step 1: Remove "Life Ladder Numeric" column
        int removeIndex = data.attribute("Life Ladder Numeric") != null
                ? data.attribute("Life Ladder Numeric").index()
                : -1;

        if (removeIndex != -1) {
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(new int[]{removeIndex});
            remove.setInvertSelection(false);
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
            System.out.println("Removed 'Life Ladder Numeric' before training.");
        } else {
            System.out.println("Attribute 'Life Ladder Numeric' not found — skipping removal.");
        }

        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute att = data.attribute(i);
            String type = att.isNumeric() ? "Numeric" :
                    att.isNominal() ? "Nominal" :
                            att.isString() ? "String" : "Other";
            System.out.println("  - [" + i + "] " + att.name() + " (" + type + ")");
        }

        // Step 2: Train RandomForest
        data.setClassIndex(classIndex >= data.numAttributes() ? data.numAttributes() - 1 : classIndex);
        RandomForest rf = new RandomForest();
        rf.setNumIterations(50);
        rf.buildClassifier(data);
        System.out.println("RandomForest trained with " + rf.getNumIterations() + " trees.");

        return new TrainResult(rf, data);
    }
    // You can add more: trainJ48, trainLogistic, trainSVM, etc.

    public static TrainResult trainMultilayerPerceptron(Instances data, int classIndex) throws Exception {
        // Step 1: Remove "Life Ladder Numeric" column
        int removeIndex = data.attribute("Life Ladder Numeric") != null
                ? data.attribute("Life Ladder Numeric").index()
                : -1;

        if (removeIndex != -1) {
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(new int[]{removeIndex});
            remove.setInvertSelection(false);
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
            System.out.println("Removed 'Life Ladder Numeric' before training.");
        } else {
            System.out.println("Attribute 'Life Ladder Numeric' not found — skipping removal.");
        }

        // Step 2: Train MultilayerPerceptron
        data.setClassIndex(classIndex-1);

        MultilayerPerceptron mlp = new MultilayerPerceptron();
        // Example configuration (you can tune these):
        mlp.setLearningRate(0.3);     // default 0.3
        mlp.setMomentum(0.2);         // default 0.2
        mlp.setTrainingTime(500);     // number of epochs
        mlp.setHiddenLayers("a");     // "a" = (attributes+classes)/2 hidden units

        mlp.buildClassifier(data);

        System.out.println("Multilayer Perceptron trained with "
                + mlp.getTrainingTime() + " epochs.");
        return new TrainResult(mlp, data);
    }

    public static TrainResult trainKNN(Instances data, int classIndex) throws Exception {
        // Step 1: Remove "Life Ladder Numeric" column if present
        int removeIndex = data.attribute("Life Ladder Numeric") != null
                ? data.attribute("Life Ladder Numeric").index()
                : -1;

        if (removeIndex != -1) {
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(new int[]{removeIndex});
            remove.setInvertSelection(false);
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
            System.out.println("Removed 'Life Ladder Numeric' before training.");
        } else {
            System.out.println("Attribute 'Life Ladder Numeric' not found — skipping removal.");
        }

        // Step 2: Set class index
        data.setClassIndex(classIndex >= data.numAttributes() ? data.numAttributes() - 1 : classIndex);

        // Step 3: Configure and train KNN (IBk)
        IBk knn = new IBk();
        knn.setKNN(5); // you can tune k, e.g. 3, 5, 7
        knn.buildClassifier(data);

        System.out.println("KNN (IBk) trained with k=" + knn.getKNN());

        return new TrainResult(knn, data);
    }
}
