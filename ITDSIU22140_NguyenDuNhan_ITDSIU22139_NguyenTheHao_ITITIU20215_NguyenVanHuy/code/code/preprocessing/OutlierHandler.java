package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.preprocessing;

import ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.analysis.StatisticsUtils;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

public class OutlierHandler {

    private static Map<String, Quartiles> cachedQuartiles = null;

    public static Map<String, Quartiles> computeQuartilesForAllNumeric(Instances data) {
        Map<String, Quartiles> map = new LinkedHashMap<>();
        for (int a = 0; a < data.numAttributes(); a++) {
            Attribute att = data.attribute(a);
            if (!att.isNumeric()) continue;
            List<Double> vals = new ArrayList<>();
            for (int r = 0; r < data.numInstances(); r++) {
                Instance inst = data.instance(r);
                if (!inst.isMissing(a)) vals.add(inst.value(a));
            }
            if (vals.isEmpty()) continue;
            Collections.sort(vals);
            double q1 = StatisticsUtils.percentile(vals, 25);
            double q3 = StatisticsUtils.percentile(vals, 75);
            double iqr = q3 - q1;
            double lw = q1 - 1.5 * iqr;
            double uw = q3 + 1.5 * iqr;
            map.put(att.name(), new Quartiles(q1, q3, iqr, lw, uw));
        }
        cachedQuartiles = map;
        return map;
    }

    public static void computeAndReportQuartiles(Instances data) {
        Map<String, Quartiles> qmap = computeQuartilesForAllNumeric(data);
        System.out.println("Whiskers and quartiles:");
        for (Map.Entry<String, Quartiles> e : qmap.entrySet()) {
            Quartiles q = e.getValue();
            System.out.println("  - " + e.getKey() + " | Q1=" + q.q1 + " | Q3=" + q.q3 +
                    " | IQR=" + q.iqr + " | lw=" + q.lw + " | uw=" + q.uw);
        }
    }

    public static void winsorizeByWhiskers(Instances data) {
        Map<String, Quartiles> qmap = cachedQuartiles != null ? cachedQuartiles : computeQuartilesForAllNumeric(data);
        for (int a = 0; a < data.numAttributes(); a++) {
            Attribute att = data.attribute(a);
            if (!att.isNumeric()) continue;
            Quartiles q = qmap.get(att.name());
            if (q == null) continue;
            int low = 0, high = 0;
            for (int r = 0; r < data.numInstances(); r++) {
                Instance inst = data.instance(r);
                if (inst.isMissing(a)) continue;
                double v = inst.value(a);
                if (v < q.lw) { inst.setValue(a, q.lw); low++; }
                else if (v > q.uw) { inst.setValue(a, q.uw); high++; }
            }
            if (low + high > 0) {
                System.out.println("Winsorized " + att.name() + " (low=" + low + ", high=" + high + ")");
            }
        }
    }

    public static class Quartiles {
        public final double q1, q3, iqr, lw, uw;
        public Quartiles(double q1, double q3, double iqr, double lw, double uw) {
            this.q1 = q1; this.q3 = q3; this.iqr = iqr; this.lw = lw; this.uw = uw;
        }
    }
}