package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.modeling;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

public class ModelEvaluator {

    public static void crossValidate(Classifier model, Instances data, int folds) throws Exception {
        data.setClassIndex(data.classIndex() < 0 ? data.numAttributes() - 1 : data.classIndex());
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, folds, new Random(42));

        System.out.println("Cross-validation (" + folds + " folds):");
        System.out.println("  - Accuracy: " + String.format("%.4f", (1.0 - eval.errorRate()) * 100) + "%");

        if (data.classAttribute().isNominal()) {
            System.out.println("  - Kappa: " + String.format("%.4f", eval.kappa()));
            System.out.println("  - Macro F1: " + String.format("%.4f", macroF1(eval, data)));
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } else {
            System.out.println("  - Regression mode: skipping Kappa and F1");
            System.out.println("  - MAE: " + String.format("%.4f", eval.meanAbsoluteError()));
            System.out.println("  - RMSE: " + String.format("%.4f", eval.rootMeanSquaredError()));
        }
    }

    private static double macroF1(Evaluation eval, Instances data) {
        int classes = data.numClasses();
        double sum = 0.0;
        for (int i = 0; i < classes; i++) sum += eval.fMeasure(i);
        return sum / classes;
    }
}