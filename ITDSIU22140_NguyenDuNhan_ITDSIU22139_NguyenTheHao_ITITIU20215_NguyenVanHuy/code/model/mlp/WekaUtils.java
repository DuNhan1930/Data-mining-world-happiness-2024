package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaUtils {

    public static Instances loadData(String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        // Set class index to the last attribute if not set
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    // Converts input attributes (excluding class) to double array
    public static double[] instanceToInput(Instance inst) {
        double[] input = new double[inst.numAttributes() - 1];
        for (int i = 0; i < input.length; i++) {
            input[i] = inst.value(i);
        }
        return input;
    }

    // Converts class index to One-Hot vector
    // e.g., Class 0 -> [1, 0, 0], Class 1 -> [0, 1, 0]
    public static double[] targetToOneHot(Instance inst) {
        int numClasses = inst.numClasses();
        double[] oneHot = new double[numClasses];
        int classIdx = (int) inst.classValue();
        oneHot[classIdx] = 1.0;
        return oneHot;
    }
}