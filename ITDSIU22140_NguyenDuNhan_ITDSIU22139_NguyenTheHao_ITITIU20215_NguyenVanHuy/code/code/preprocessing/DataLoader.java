package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.preprocessing;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;

import java.io.File;

public class DataLoader {
    public static Instances load(String path) throws Exception {
        Instances data;

        if (path.toLowerCase().endsWith(".csv")) {
            // Load CSV
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(path));
            Instances csvData = loader.getDataSet();

            // Save to ARFF
            String arffPath = path.replace(".csv", ".arff");
            ArffSaver saver = new ArffSaver();
            saver.setInstances(csvData);
            saver.setFile(new File(arffPath));
            saver.writeBatch();
            System.out.println("CSV converted and saved as ARFF: " + arffPath);

            // Reload from ARFF so we use ARFF for modeling
            DataSource source = new DataSource(arffPath);
            data = source.getDataSet();
            System.out.println("Reloaded ARFF for modeling: " + arffPath);

        } else if (path.toLowerCase().endsWith(".arff")) {
            // Load ARFF directly
            DataSource source = new DataSource(path);
            data = source.getDataSet();
            System.out.println("Loaded ARFF file: " + path);

        } else {
            throw new IllegalArgumentException("Unsupported file format: " + path);
        }

        // Ensure class index is set (default: last column)
        if (data.classIndex() < 0 && data.numAttributes() > 0) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        return data;
    }
}