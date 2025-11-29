package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.preprocessing;

import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.analysis.StatisticsUtils;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;

public class MissingValueHandler {

    public static void imputeMedian(Instances data, List<String> columns) {
        for (String col : columns) {
            Attribute att = data.attribute(col);
            if (att == null) {
                System.out.println("Warning: column not found for median: " + col);
                continue;
            }
            if (!att.isNumeric()) {
                System.out.println("Warning: non-numeric median column: " + col);
                continue;
            }
            double median = StatisticsUtils.medianOfAttribute(data, att.index());
            for (int r = 0; r < data.numInstances(); r++) {
                Instance inst = data.instance(r);
                if (inst.isMissing(att.index())) {
                    inst.setValue(att.index(), median);
                }
            }
            System.out.println("Median-imputed: " + col + " (median=" + median + ")");
        }
    }

    public static void imputeKNN(Instances data, List<String> columns, int k) {
        List<Integer> numericAtts = new ArrayList<>();
        for (int a = 0; a < data.numAttributes(); a++) {
            if (data.attribute(a).isNumeric()) numericAtts.add(a);
        }

        for (String col : columns) {
            Attribute att = data.attribute(col);
            if (att == null) {
                System.out.println("Warning: column not found for KNN: " + col);
                continue;
            }
            if (!att.isNumeric()) {
                System.out.println("Warning: non-numeric KNN column: " + col);
                continue;
            }
            int targetIdx = att.index();
            List<Integer> features = numericAtts.stream().filter(i -> i != targetIdx).collect(Collectors.toList());

            int imputed = 0;
            for (int r = 0; r < data.numInstances(); r++) {
                Instance inst = data.instance(r);
                if (!inst.isMissing(targetIdx)) continue;

                List<Neighbor> neighbors = new ArrayList<>();
                for (int s = 0; s < data.numInstances(); s++) {
                    if (s == r) continue;
                    Instance other = data.instance(s);
                    if (other.isMissing(targetIdx)) continue;
                    Double dist = euclideanDistance(inst, other, features);
                    if (dist == null) continue;
                    neighbors.add(new Neighbor(s, dist, other.value(targetIdx)));
                }

                if (neighbors.isEmpty()) {
                    System.out.println("Warning: no neighbors for row " + r + " in " + col);
                    continue;
                }

                neighbors.sort(Comparator.comparingDouble(n -> n.distance));
                int take = Math.min(k, neighbors.size());
                List<Double> kVals = new ArrayList<>(take);
                for (int i = 0; i < take; i++) kVals.add(neighbors.get(i).targetValue);

                double value = StatisticsUtils.medianOfList(kVals);
                inst.setValue(targetIdx, value);
                imputed++;
            }
            System.out.println("KNN-imputed: " + col + " (k=" + k + ", imputed=" + imputed + ")");
        }
    }

    private static Double euclideanDistance(Instance a, Instance b, List<Integer> features) {
        double sum = 0.0;
        int used = 0;
        for (int idx : features) {
            if (a.isMissing(idx) || b.isMissing(idx)) continue;
            double da = a.value(idx);
            double db = b.value(idx);
            sum += (da - db) * (da - db);
            used++;
        }
        if (used == 0) return null;
        return Math.sqrt(sum);
    }

    private static class Neighbor {
        int rowIndex;
        double distance;
        double targetValue;
        Neighbor(int r, double d, double v) { rowIndex = r; distance = d; targetValue = v; }
    }
}
