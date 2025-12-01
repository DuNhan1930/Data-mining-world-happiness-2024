package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.preprocessing;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MissingValueHandler {

    /**
     * Impute missing numeric values in the given columns using median.
     * EN: Like df[col] = df[col].fillna(df[col].median())
     * VI: Giống fillna(median) cho từng cột số.
     */
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

            int idx = att.index();
            List<Double> vals = new ArrayList<>();
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                if (!inst.isMissing(idx)) {
                    vals.add(inst.value(idx));
                }
            }
            if (vals.isEmpty()) {
                System.out.println("Warning: all values missing for " + col);
                continue;
            }

            Collections.sort(vals);
            double median = medianOfList(vals);

            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                if (inst.isMissing(idx)) {
                    inst.setValue(idx, median);
                }
            }

            System.out.println("Median-imputed: " + col + " (median=" + median + ")");
        }
    }

    private static double medianOfList(List<Double> sorted) {
        int n = sorted.size();
        if (n == 0) return Double.NaN;
        if (n % 2 == 1) {
            return sorted.get(n / 2);
        } else {
            return (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2.0;
        }
    }
}
