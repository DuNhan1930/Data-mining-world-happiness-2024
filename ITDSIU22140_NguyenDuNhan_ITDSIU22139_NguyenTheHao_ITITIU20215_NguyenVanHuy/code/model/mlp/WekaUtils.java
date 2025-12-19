package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.model.mlp;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.unsupervised.attribute.Normalize;

public class WekaUtils {

    public static Instances loadData(String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();

        Attribute classAttr = data.attribute("Life Ladder Category");
        if (classAttr != null) {
            data.setClassIndex(classAttr.index());
            System.out.println("Found Class Attribute at index: " + classAttr.index());
        } else {
            throw new Exception("Error: Could not find attribute 'Life Ladder Category' in ARFF file.");
        }
        return data;
    }

    public static double[] instanceToInput(Instance inst) {
        // Input size = Tổng cột - 1 (cột class)
        double[] input = new double[inst.numAttributes() - 1];
        int k = 0;
        int classIdx = inst.classIndex();

        for (int i = 0; i < inst.numAttributes(); i++) {
            if (i == classIdx) continue; // Nhảy qua cột Class nằm giữa
            input[k++] = inst.value(i);
        }
        return input;
    }

    public static double[] targetToOneHot(Instance inst) {
        int numClasses = inst.numClasses();
        double[] oneHot = new double[numClasses];
        oneHot[(int) inst.classValue()] = 1.0;
        return oneHot;
    }

    public static Instances normalizeData(Instances data) throws Exception {
        Normalize norm = new Normalize();
        norm.setInputFormat(data);
        return Filter.useFilter(data, norm);
    }

    public static Instances balanceData(Instances data) throws Exception {
        ClassBalancer balancer = new ClassBalancer();
        balancer.setInputFormat(data);
        return Filter.useFilter(data, balancer);
    }
}