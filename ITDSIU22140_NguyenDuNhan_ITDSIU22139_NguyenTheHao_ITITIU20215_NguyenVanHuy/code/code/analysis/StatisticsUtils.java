package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.analysis;

import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class StatisticsUtils {

    public static double medianOfAttribute(Instances data, int attIndex) {
        List<Double> vals = new ArrayList<>();
        for (int r = 0; r < data.numInstances(); r++) {
            Instance inst = data.instance(r);
            if (!inst.isMissing(attIndex)) {
                vals.add(inst.value(attIndex));
            }
        }
        if (vals.isEmpty()) return Double.NaN;
        Collections.sort(vals);
        int n = vals.size();
        if (n % 2 == 1) return vals.get(n / 2);
        return (vals.get(n / 2 - 1) + vals.get(n / 2)) / 2.0;
    }

    public static double medianOfList(List<Double> vals) {
        Collections.sort(vals);
        int n = vals.size();
        if (n == 0) return Double.NaN;
        if (n % 2 == 1) return vals.get(n / 2);
        return (vals.get(n / 2 - 1) + vals.get(n / 2)) / 2.0;
    }

    public static double percentile(List<Double> sortedVals, double p) {
        int n = sortedVals.size();
        if (n == 0) return Double.NaN;
        if (n == 1) return sortedVals.get(0);
        double pos = (p / 100.0) * (n - 1);
        int lower = (int) Math.floor(pos);
        int upper = (int) Math.ceil(pos);
        if (lower == upper) return sortedVals.get(lower);
        double weight = pos - lower;
        return sortedVals.get(lower) * (1 - weight) + sortedVals.get(upper) * weight;
    }
}