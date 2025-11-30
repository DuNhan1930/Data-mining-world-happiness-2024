package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code;

// Weka core
import weka.core.Instances;

// Pipeline helper classes
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.preprocessing.*;
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.analysis.*;
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.modeling.*;
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.export.*;

import java.util.List;

public class MainPipeline {
    public static void main(String[] args) throws Exception {
        // 1) Load
        System.out.println("Step 1: Loading data...");
        Instances data = DataLoader.load(
                "ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy/code/data/World Happiness Report 2024.csv"
        );
        System.out.println();

        // 2) Profile (shape, info, missing counts, duplicates)
        System.out.println("Step 2: Profiling...");
        DataProfiler.profile(data);
        System.out.println();

        // 3) Data Transformer
        System.out.println("Step 3: Data transforming...");
        // Remove unnecessary columns
        data = DataTransformer.removeUnnecessaryColumns(data);
        // Add Life Ladder Category
        data = DataTransformer.addLifeLadderCategory(data);
        // Rename and Reorder Life Ladder Numeric
        data = DataTransformer.renameAndReorderLifeLadder(data);
        // Set class index to new category
        data.setClassIndex(data.attribute("Life Ladder Category").index());
        System.out.println();

        // 4) Missing values: median for some, KNN for others
        System.out.println("Step 4: Imputing missing values...");
        MissingValueHandler.imputeMedian(data, List.of(
                "Log GDP per capita",
                "Social support",
                "Positive affect",
                "Negative affect"
        ));
        MissingValueHandler.imputeKNN(data, List.of(
                "Healthy life expectancy at birth",
                "Freedom to make life choices",
                "Generosity",
                "Perceptions of corruption"
        ), 5);
        System.out.println();

        // 5) Outliers: compute whiskers and winsorize
        System.out.println("Step 5: Handling outliers...");
        OutlierHandler.computeAndReportQuartiles(data);
        OutlierHandler.winsorizeByWhiskers(data);
        System.out.println();

        // 6) Dataset preparation (identify attributes + train/test split)
        System.out.println("Step 6: Preparing train, test data...");
        DatasetPreparer.printAttributes(data);
        Instances[] split = DatasetPreparer.trainTestSplit(data, 0.8); // 80% train, 20% test
        Instances train = split[0];
        Instances test = split[1];
        System.out.println();


        System.out.println("Step 7: Classifier training and Showing results... ");

        // 7.1) Train Multilayer Perceptron
        System.out.println("Model 1: MultilayerPerception Traning...");
        ClassifierTrainer.TrainResult mlpResult = ClassifierTrainer.trainMultilayerPerceptron(train, train.numAttributes() - 1);
        System.out.println("Model 1: MultilayerPerception Showing results...");
        ModelEvaluator.crossValidate(mlpResult.model, test, 10);

        // 7.2) Train RandomForest
        System.out.println("Model 2: RandomForest Traning...");
        ClassifierTrainer.TrainResult rfResult = ClassifierTrainer.trainRandomForest(train, train.numAttributes() - 1);
        System.out.println("Model 2: RandomForest Showing results...");
        ModelEvaluator.crossValidate(rfResult.model, test, 10);
        System.out.println();

        // 8) Save cleaned dataset
        DataSaver.saveArff(data,
                "ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy/code/data/World Happiness Report 2024 cleaned.arff"
        );

        System.out.println("Pipeline complete.");
    }
}
