package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.preprocessing;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;

public class DataLoader {
    public static Instances load(String path) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        Instances data = loader.getDataSet();

        // If you have a class column (target), set it here
        // Example: last column is the class
        if (data.classIndex() < 0 && data.numAttributes() > 0) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }
}