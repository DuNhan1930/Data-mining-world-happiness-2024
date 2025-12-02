package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.analysis;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class ExploreData {

    // Path to the RAW CSV file.
    private static final String CSV_PATH =
            "ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy/code/data/World Happiness Report 2024.csv";

    public static void main(String[] args) throws Exception {

        // 1. Load raw CSV as Instances
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(CSV_PATH));
        Instances data = loader.getDataSet();

        System.out.println("Loaded RAW CSV: " + CSV_PATH);
        System.out.println("Relation name after load: " + data.relationName());

        // 2. Define numeric columns from RAW CSV
        List<String> numericCols = Arrays.asList(
                "year",
                "Life Ladder",
                "Log GDP per capita",
                "Social support",
                "Healthy life expectancy at birth",
                "Freedom to make life choices",
                "Generosity",
                "Perceptions of corruption",
                "Positive affect",
                "Negative affect"
        );

        // 3. Run EDA
        EDA.printDatasetInfo(data);
        EDA.printMissingValues(data);                 // null counts
        EDA.printNumericSummaries(data, numericCols); // mean, median, std, ...
        EDA.printOutlierStats(data, numericCols);     // outlier counts (justify robust scaler)
        EDA.printCorrelationMatrix(data, numericCols);// correlation matrix
        EDA.printSkewness(data, numericCols);         // skewness (check asymmetry)
    }
}
