package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.analysis;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Utility class for basic EDA on a Weka Instances dataset.
 */
public class EDA {

    // ============================
    // 1. Basic dataset information
    // ============================
    public static void printDatasetInfo(Instances data) {
        System.out.println("=== DATASET INFO ===");
        System.out.println("Relation name : " + data.relationName());
        System.out.println("Num instances : " + data.numInstances());
        System.out.println("Num attributes: " + data.numAttributes());
        System.out.println("Class index   : " + data.classIndex()
                + " (" + (data.classIndex() >= 0 ? data.classAttribute().name() : "no class set") + ")");
        System.out.println();
    }

    // ==========================================
    // 2. Missing values per attribute (null count)
    // ==========================================
    public static void printMissingValues(Instances data) {
        System.out.println("=== MISSING VALUES PER ATTRIBUTE ===");
        int n = data.numInstances();
        for (int a = 0; a < data.numAttributes(); a++) {
            Attribute att = data.attribute(a);
            int missing = 0;
            for (int i = 0; i < n; i++) {
                Instance inst = data.instance(i);
                if (inst.isMissing(a)) missing++;
            }
            double pct = (n == 0) ? 0.0 : (100.0 * missing / n);
            System.out.printf("Attribute %-35s : missing=%4d (%.2f%%)%n",
                    att.name(), missing, pct);
        }
        System.out.println();
    }

    // ==========================================
    // 3. Numeric summaries: min, max, mean, median, std
    // ==========================================
    public static void printNumericSummaries(Instances data, List<String> numericCols) {
        System.out.println("=== NUMERIC SUMMARIES (min, max, mean, median, std) ===");
        for (String col : numericCols) {
            Attribute att = data.attribute(col);
            if (att == null) {
                System.out.println("Warning: numeric col not found: " + col);
                continue;
            }
            if (!att.isNumeric()) {
                System.out.println("Warning: col is not numeric: " + col);
                continue;
            }

            List<Double> vals = collectNonMissingValues(data, att.index());
            if (vals.isEmpty()) {
                System.out.println("No data for: " + col);
                continue;
            }

            Collections.sort(vals);
            double min = vals.get(0);
            double max = vals.get(vals.size() - 1);
            double mean = mean(vals);
            double median = median(vals);
            double std = stdDev(vals, mean);

            System.out.printf("Attribute %-35s : min=%10.4f | max=%10.4f | mean=%10.4f | median=%10.4f | std=%10.4f%n",
                    col, min, max, mean, median, std);
        }
        System.out.println();
    }

    // ==========================================
    // 4. Correlation matrix (Pearson) for selected numeric columns
    // ==========================================
    public static void printCorrelationMatrix(Instances data, List<String> numericCols) {
        System.out.println("=== CORRELATION MATRIX (Pearson) ===");
        int m = numericCols.size();
        int[] attIdx = new int[m];

        for (int i = 0; i < m; i++) {
            String col = numericCols.get(i);
            Attribute att = data.attribute(col);
            if (att == null || !att.isNumeric()) {
                System.out.println("Warning: cannot correlate, invalid numeric col: " + col);
                return;
            }
            attIdx[i] = att.index();
        }
        double[][] corr = new double[m][m];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                if (i == j) {
                    corr[i][j] = 1.0;
                } else if (j < i) {
                    corr[i][j] = corr[j][i]; // symmetric
                } else {
                    corr[i][j] = pearsonCorr(data, attIdx[i], attIdx[j]);
                }
            }
        }
        // print header
        System.out.printf("%-28s", "");
        for (int j = 0; j < m; j++) {
            System.out.printf("%-12s", numericCols.get(j));
        }
        System.out.println();
        // print rows
        for (int i = 0; i < m; i++) {
            System.out.printf("%-28s", numericCols.get(i));
            for (int j = 0; j < m; j++) {
                System.out.printf("%-12.3f", corr[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }

    // ==========================================
    // 5. Outlier statistics via Tukey's fences
    // ==========================================
    public static void printOutlierStats(Instances data, List<String> numericCols) {
        System.out.println("=== OUTLIER STATS (Tukey fences) ===");

        for (String col : numericCols) {
            Attribute att = data.attribute(col);
            if (att == null) {
                System.out.println("Warning: numeric col not found: " + col);
                continue;
            }
            if (!att.isNumeric()) {
                System.out.println("Warning: col is not numeric: " + col);
                continue;
            }

            List<Double> vals = collectNonMissingValues(data, att.index());
            if (vals.isEmpty()) {
                System.out.println("No data for: " + col);
                continue;
            }
            Collections.sort(vals);
            double q1 = percentile(vals, 25.0);
            double q3 = percentile(vals, 75.0);
            double iqr = q3 - q1;
            double lw = q1 - 1.5 * iqr;
            double uw = q3 + 1.5 * iqr;

            int lowCount = 0;
            int highCount = 0;

            for (double v : vals) {
                if (v < lw) lowCount++;
                else if (v > uw) highCount++;
            }

            System.out.printf("Attribute %-35s : Q1=%8.4f | Q3=%8.4f | IQR=%8.4f | lw=%8.4f | uw=%8.4f | lowOut=%4d | highOut=%4d%n",
                    col, q1, q3, iqr, lw, uw, lowCount, highCount);
        }
        System.out.println();
    }

    // ==========================================
    // 6. Skewness per numeric attribute
    // ==========================================
    public static void printSkewness(Instances data, List<String> numericCols) {
        System.out.println("=== SKEWNESS PER ATTRIBUTE ===");
        System.out.println("Note: |skew| < 0.5 ≈ symmetric, 0.5–1 ≈ moderate, > 1 = highly skewed");
        System.out.println();

        for (String col : numericCols) {
            Attribute att = data.attribute(col);
            if (att == null) {
                System.out.println("Warning: numeric col not found: " + col);
                continue;
            }
            if (!att.isNumeric()) {
                System.out.println("Warning: col is not numeric: " + col);
                continue;
            }

            List<Double> vals = collectNonMissingValues(data, att.index());
            if (vals.isEmpty()) {
                System.out.println("No data for: " + col);
                continue;
            }

            double sk = skewness(vals);
            double absSk = Math.abs(sk);
            String level;
            if (Double.isNaN(sk)) {
                level = "NA (not enough data)";
            } else if (absSk < 0.5) {
                level = "≈ symmetric";
            } else if (absSk < 1.0) {
                level = "moderately skewed";
            } else {
                level = "highly skewed";
            }

            System.out.printf("Attribute %-35s : skewness=%8.4f → %s%n",
                    col, sk, level);
        }
        System.out.println();
    }

    // ===========================
    // Helper methods
    // ===========================
    private static List<Double> collectNonMissingValues(Instances data, int attIndex) {
        List<Double> vals = new ArrayList<>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            if (!inst.isMissing(attIndex)) {
                vals.add(inst.value(attIndex));
            }
        }
        return vals;
    }

    private static double mean(List<Double> vals) {
        if (vals.isEmpty()) return Double.NaN;
        double sum = 0.0;
        for (double v : vals) sum += v;
        return sum / vals.size();
    }

    private static double median(List<Double> sorted) {
        int n = sorted.size();
        if (n == 0) return Double.NaN;
        if (n % 2 == 1) {
            return sorted.get(n / 2);
        } else {
            return (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2.0;
        }
    }

    private static double stdDev(List<Double> vals, double mean) {
        if (vals.isEmpty()) return Double.NaN;
        double sumSq = 0.0;
        for (double v : vals) {
            double d = v - mean;
            sumSq += d * d;
        }
        return Math.sqrt(sumSq / vals.size());
    }

    /**
     * Pearson correlation between two attributes (by index), skipping rows with missing values.
     */
    private static double pearsonCorr(Instances data, int idx1, int idx2) {
        double sumX = 0.0, sumY = 0.0;
        double sumXX = 0.0, sumYY = 0.0, sumXY = 0.0;
        int n = 0;

        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            if (inst.isMissing(idx1) || inst.isMissing(idx2)) continue;

            double x = inst.value(idx1);
            double y = inst.value(idx2);

            sumX += x;
            sumY += y;
            sumXX += x * x;
            sumYY += y * y;
            sumXY += x * y;
            n++;
        }

        if (n < 2) return Double.NaN;

        double meanX = sumX / n;
        double meanY = sumY / n;

        double covXY = sumXY - n * meanX * meanY;
        double varX  = sumXX - n * meanX * meanX;
        double varY  = sumYY - n * meanY * meanY;

        if (varX <= 0 || varY <= 0) return 0.0;

        return covXY / Math.sqrt(varX * varY);
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

    /**
     * Skewness of a list of numeric values.
     * Measures asymmetry of the distribution.
     * skewness ≈ 0      → symmetric
     * skewness > 0      → tail on the right (right-skewed)
     * skewness < 0      → tail on the left (left-skewed)
     */
    private static double skewness(List<Double> vals) {
        int n = vals.size();
        if (n < 2) {
            return Double.NaN;
        }

        double mean = mean(vals);
        double sum2 = 0.0; // Σ (x - mean)^2
        double sum3 = 0.0; // Σ (x - mean)^3

        for (double v : vals) {
            double d = v - mean;
            double d2 = d * d;
            sum2 += d2;
            sum3 += d2 * d; // (x - mean)^3
        }

        double m2 = sum2 / n;  // 2nd central moment
        double m3 = sum3 / n;  // 3rd central moment

        if (m2 == 0.0) {
            // all values identical → no variance → skewness = 0
            return 0.0;
        }

        // Fisher-Pearson coefficient of skewness:
        // skew = m3 / (m2^(3/2))
        return m3 / Math.pow(m2, 1.5);
    }
}
