package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.preprocessing;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class RobustScalerHandler {

    /**
     * Robust scale numeric columns using:
     *   x_scaled = (x - median) / IQR
     * where IQR = Q3 - Q1.
     */
    public static void robustScale(Instances data, List<String> numericCols) {
        for (String col : numericCols) {
            Attribute att = data.attribute(col);
            if (att == null) {
                System.out.println("Warning: column not found for robust scaling: " + col);
                continue;
            }
            if (!att.isNumeric()) {
                System.out.println("Warning: non-numeric robust scaling column: " + col);
                continue;
            }

            int idx = att.index();

            // Collect non-missing values
            List<Double> vals = new ArrayList<>();
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                if (!inst.isMissing(idx)) {
                    vals.add(inst.value(idx));
                }
            }
            if (vals.isEmpty()) {
                System.out.println("Warning: no numeric data for " + col);
                continue;
            }

            // Sort for median + percentiles
            Collections.sort(vals);
            double median = medianOfList(vals);
            double q1 = percentile(vals, 25.0);
            double q3 = percentile(vals, 75.0);
            double iqr = q3 - q1;

            if (Double.isNaN(median) || Double.isNaN(iqr) || iqr == 0.0) {
                System.out.println("Warning: invalid IQR (" + iqr + ") for " + col + ", skip robust scaling.");
                continue;
            }

            // Apply (x - median) / IQR to all non-missing values
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                if (inst.isMissing(idx)) continue;
                double x = inst.value(idx);
                double scaled = (x - median) / iqr;
                inst.setValue(idx, scaled);
            }

            System.out.println("Robust scaled " + col + " (median=" + median + ", IQR=" + iqr + ")");
        }
    }

    // ----- helpers -----

    private static double medianOfList(List<Double> sorted) {
        int n = sorted.size();
        if (n == 0) return Double.NaN;
        if (n % 2 == 1) {
            return sorted.get(n / 2);
        } else {
            return (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2.0;
        }
    }

    private static double percentile(List<Double> sorted, double p) {
        int n = sorted.size();
        if (n == 0) return Double.NaN;
        if (p <= 0) return sorted.get(0);
        if (p >= 100) return sorted.get(n - 1);

        double pos = (p / 100.0) * (n - 1);
        int lower = (int) Math.floor(pos);
        int upper = (int) Math.ceil(pos);
        double weight = pos - lower;

        double lowerVal = sorted.get(lower);
        double upperVal = sorted.get(upper);
        return lowerVal + weight * (upperVal - lowerVal);
    }
}
