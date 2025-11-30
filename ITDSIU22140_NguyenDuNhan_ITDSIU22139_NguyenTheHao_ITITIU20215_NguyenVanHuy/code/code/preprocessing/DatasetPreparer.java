package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.preprocessing;

import weka.core.Instances;
import java.util.Random;

public class DatasetPreparer {
    // Split dataset into train/test sets
    public static Instances[] trainTestSplit(Instances data, double trainPercent) throws Exception {
        data.randomize(new Random(42));
        int trainSize = (int) Math.round(data.numInstances() * trainPercent);
        int testSize = data.numInstances() - trainSize;

        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        System.out.println("Dataset split: " + trainSize + " train / " + testSize + " test");
        return new Instances[]{train, test};
    }

    // Print attribute names for identification
    public static void printAttributes(Instances data) {
        System.out.println("Attributes in dataset:");
        for (int i = 0; i < data.numAttributes(); i++) {
            System.out.println(" - " + data.attribute(i).name());
        }
    }
}
