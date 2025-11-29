package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code;

// Weka core
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.preprocessing.DataTransformer;
import weka.core.Instances;

// Pipeline helper classes
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.preprocessing.DataLoader;
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.analysis.DataProfiler;
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.export.DataSaver;
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.modeling.ClassifierTrainer;
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.modeling.ModelEvaluator;
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.preprocessing.MissingValueHandler;
import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.preprocessing.OutlierHandler;

import java.util.List;

public class MainPipeline {
    public static void main(String[] args) throws Exception {
        // 1) Load
        System.out.println("Step 1: Loading data...");
        Instances data = DataLoader.load(
                "C:/Users/Admin/IdeaProjects/Data-mining-world-happiness-2024/ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy/code/data/World Happiness Report 2024.csv"
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

        System.out.println("Step 6: Classifier training and Showing results... ");
        // Train RandomForest
        System.out.println("Model 1: RandomForest Traning...");
        ClassifierTrainer.TrainResult rfResult = ClassifierTrainer.trainRandomForest(data, data.numAttributes() - 1);
        System.out.println("Model 1: RandomForest Showing results...");
        ModelEvaluator.crossValidate(rfResult.model, data, 10);
        System.out.println();

        // Train Multilayer Perceptron
        System.out.println("Model 2: MultilayerPerception Traning...");
        ClassifierTrainer.TrainResult mlpResult = ClassifierTrainer.trainMultilayerPerceptron(data, data.numAttributes() - 1);
        System.out.println("Model 2: MultilayerPerception Showing results...");
        ModelEvaluator.crossValidate(mlpResult.model, data, 10);

        // 8) Save cleaned dataset
        DataSaver.saveArff(data,
                "C:/Users/Admin/IdeaProjects/Data-mining-world-happiness-2024/ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy/code/data/World_Happiness_2024_cleaned.arff"
        );

        System.out.println("Pipeline complete.");
    }
}
