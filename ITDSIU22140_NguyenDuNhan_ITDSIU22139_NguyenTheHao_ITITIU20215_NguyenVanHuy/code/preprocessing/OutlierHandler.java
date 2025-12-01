package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.preprocessing;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class OutlierHandler {

    /**
     * Winsorize given numeric columns using Tukey's whiskers:
     * lw = Q1 - 1.5 * IQR, uw = Q3 + 1.5 * IQR
     * EN: Values < lw are clipped to lw, > uw are clipped to uw.
     * VI: Các giá trị nhỏ hơn lw sẽ bị "cắt" lên lw, lớn hơn uw bị "cắt" xuống uw.
     */
    public static void winsorizeByWhiskers(Instances data, List<String> numericCols) {
        for (String col : numericCols) {
            Attribute att = data.attribute(col);
            if (att == null) {
                System.out.println("Warning: column not found for winsorization: " + col);
                continue;
            }
            if (!att.isNumeric()) {
                System.out.println("Warning: non-numeric winsor column: " + col);
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
                System.out.println("Warning: no numeric data for " + col);
                continue;
            }

            Collections.sort(vals);
            double q1 = percentile(vals, 25.0);
            double q3 = percentile(vals, 75.0);
            double iqr = q3 - q1;
            double lw  = q1 - 1.5 * iqr;
            double uw  = q3 + 1.5 * iqr;

            int lowCount = 0;
            int highCount = 0;
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                if (inst.isMissing(idx)) continue;
                double v = inst.value(idx);
                if (v < lw) {
                    inst.setValue(idx, lw);
                    lowCount++;
                } else if (v > uw) {
                    inst.setValue(idx, uw);
                    highCount++;
                }
            }

            if (lowCount + highCount > 0) {
                System.out.println("Winsorized " + col + " (low=" + lowCount + ", high=" + highCount +
                        ", lw=" + lw + ", uw=" + uw + ")");
            }
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
        return sorted.get(lower) + weight * (sorted.get(upper) - sorted.get(lower));
    }
}
