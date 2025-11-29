package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.export;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.File;

public class DataSaver {
    public static void saveArff(Instances data, String path) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(path));
        saver.writeBatch();
        System.out.println("Saved ARFF: " + path);
    }
}