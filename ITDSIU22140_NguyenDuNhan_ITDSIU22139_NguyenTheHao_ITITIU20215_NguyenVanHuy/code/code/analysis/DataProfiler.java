package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.analysis;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashSet;
import java.util.Set;

public class DataProfiler {

    public static void profile(Instances data) {
        printShape(data);
        printInfo(data);
        printMissingCounts(data);
        printDuplicateCount(data);
    }

    public static void printShape(Instances data) {
        System.out.println("Shape: (" + data.numInstances() + ", " + data.numAttributes() + ")");
    }

    public static void printInfo(Instances data) {
        System.out.println("Info:");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute att = data.attribute(i);
            String type = att.isNumeric() ? "numeric" : (att.isNominal() ? "nominal" : (att.isString() ? "string" : "other"));
            int missing = countMissingInAttribute(data, att.index());
            System.out.println("  - " + att.name() + " | type=" + type + " | missing=" + missing);
        }
    }

    public static void printMissingCounts(Instances data) {
        System.out.println("Check Null:");
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute att = data.attribute(i);
            int missing = countMissingInAttribute(data, i);
            System.out.println("  - " + att.name() + ": " + missing);
        }
    }

    public static int countMissingInAttribute(Instances data, int attIndex) {
        int c = 0;
        for (int r = 0; r < data.numInstances(); r++) {
            if (data.instance(r).isMissing(attIndex)) c++;
        }
        return c;
    }

    public static void printDuplicateCount(Instances data) {
        Set<String> seen = new HashSet<>();
        int dup = 0;
        for (int r = 0; r < data.numInstances(); r++) {
            Instance inst = data.instance(r);
            String key = instanceKey(inst);
            if (!seen.add(key)) dup++;
        }
        System.out.println("Check Duplicate: " + dup);
    }

    private static String instanceKey(Instance inst) {
        StringBuilder sb = new StringBuilder();
        for (int a = 0; a < inst.numAttributes(); a++) {
            if (inst.isMissing(a)) {
                sb.append("NaN|");
            } else if (inst.attribute(a).isNumeric()) {
                sb.append(inst.value(a)).append("|");
            } else {
                sb.append(inst.stringValue(a)).append("|");
            }
        }
        return sb.toString();
    }
}