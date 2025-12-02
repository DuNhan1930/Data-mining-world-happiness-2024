package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.preprocessing;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class PreprocessWorldHappiness {

    private static final String INPUT_CSV  =
            "ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy/code/data/World Happiness Report 2024.csv";

    private static final String OUTPUT_ARFF =
            "ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy/code/data/World Happiness Report 2024 Preprocessed.arff";

    private static final List<String> NUM_COLS = Arrays.asList(
            "year",
            "Log GDP per capita",
            "Social support",
            "Freedom to make life choices",
            "Generosity",
            "Perceptions of corruption",
            "Positive affect",
            "Negative affect"
    );

    private static final String COUNTRY_COL = "Country name";
    private static final String LIFE_LADDER_NUM_COL = "Life Ladder";
    private static final String LIFE_LADDER_CAT_COL = "Life Ladder Category";

    public static void main(String[] args) throws Exception {

        // =====================
        // 1. Load CSV
        // =====================
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(INPUT_CSV));
        Instances data = loader.getDataSet();

        System.out.println("Loaded CSV: " + data.numInstances() + " rows, " + data.numAttributes() + " columns.");

        // =====================
        // 2. Create Life Ladder Category (bins [0,5,7,10])
        // =====================
        ensureLifeLadderCategory(data);

        // =====================
        // 3. Impute missing values by median for numeric cols
        // =====================
        MissingValueHandler.imputeMedian(data, NUM_COLS);

        // =====================
        // 4. Robust scaler numeric columns
        // =====================
        RobustScalerHandler.robustScale(data, NUM_COLS);

        // =====================
        // 5. Drop unwanted columns
        // =====================
        DropColumn.dropColumns(data, Arrays.asList(
                "Healthy life expectancy at birth",
                LIFE_LADDER_NUM_COL
        ));

        System.out.println("After drop: " + data.numAttributes() + " attributes.");

        // =====================
        // 6. One-hot encode Country name (NominalToBinary)
        // =====================
        data = EncodingHandler.oneHotEncodeNominal(data, COUNTRY_COL);

        // =====================
        // 7. Set class attribute = Life Ladder Category
        // =====================
        Attribute classAttr = data.attribute(LIFE_LADDER_CAT_COL);
        if (classAttr == null) {
            throw new IllegalStateException("Class attribute '" + LIFE_LADDER_CAT_COL + "' not found.");
        }
        data.setClass(classAttr);
        System.out.println("Class set to: " + data.classAttribute().name());

        // =====================
        // 8. Save to ARFF
        // =====================
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(OUTPUT_ARFF));
        saver.writeBatch();

        System.out.println("Saved preprocessed ARFF to: " + OUTPUT_ARFF);
        System.out.println("Done.");
    }

    /**
     * Ensure Life Ladder Category exists, constructed from Life Ladder using bins [0,5,7,10].
     */
    private static void ensureLifeLadderCategory(Instances data) {
        Attribute catAttr = data.attribute(LIFE_LADDER_CAT_COL);
        if (catAttr != null) {
            System.out.println("Life Ladder Category already exists.");
            return;
        }

        Attribute lifeLadderAttr = data.attribute(LIFE_LADDER_NUM_COL);
        if (lifeLadderAttr == null || !lifeLadderAttr.isNumeric()) {
            throw new IllegalArgumentException("Numeric attribute '" + LIFE_LADDER_NUM_COL + "' not found.");
        }

        // Create nominal attribute (Low, Medium, High, Very High)
        java.util.ArrayList<String> levels = new java.util.ArrayList<>();
        levels.add("Low");
        levels.add("Medium");
        levels.add("High");

        Attribute newCatAttr = new Attribute(LIFE_LADDER_CAT_COL, levels);
        int newIndex = data.numAttributes();
        data.insertAttributeAt(newCatAttr, newIndex);

        int lifeIdx = lifeLadderAttr.index();
        int catIdx  = data.attribute(LIFE_LADDER_CAT_COL).index();

        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            if (inst.isMissing(lifeIdx)) {
                inst.setMissing(catIdx);
                continue;
            }
            double v = inst.value(lifeIdx);
            String label;
            if (v <= 5.0) {
                label = "Low";
            } else if (v <= 7.0) {
                label = "Medium";
            } else {
                label = "High";
            }
            inst.setValue(catIdx, label);
        }

        System.out.println("Created Life Ladder Category attribute.");
    }
}
