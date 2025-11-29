package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.code.preprocessing;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;
import java.util.List;

public class DataTransformer {

    // Removes "Country name", "year", and "Healthy life expectancy at birth"
    public static Instances removeUnnecessaryColumns(Instances data) throws Exception {
        StringBuilder indices = new StringBuilder();

        for (int i = 0; i < data.numAttributes(); i++) {
            String name = data.attribute(i).name().toLowerCase();
            if (name.contains("country")
                    || name.contains("year")
                    || name.contains("healthy life expectancy")) {
                if (indices.length() > 0) indices.append(",");
                indices.append(i + 1); // Weka uses 1-based indexing
            }
        }

        if (indices.length() == 0) {
            System.out.println("No unnecessary attributes found to remove.");
            return data;
        }

        Remove remove = new Remove();
        remove.setAttributeIndices(indices.toString());
        remove.setInputFormat(data);
        Instances filtered = Filter.useFilter(data, remove);

        System.out.println("Removed attributes: " + indices);
        return filtered;
    }

    // Converts "Life Ladder" numeric to categorical "Life Ladder Category"
    public static Instances addLifeLadderCategory(Instances data) throws Exception {
        Attribute lifeLadder = data.attribute("Life Ladder");
        if (lifeLadder == null || !lifeLadder.isNumeric()) {
            System.out.println("Life Ladder column missing or not numeric.");
            return data;
        }

        // Define bins and labels
        double[] bins = {0, 2, 4, 6, 8, 10};
        String[] labels = {"Very Sad", "Sad", "Neutral", "Happy", "Very Happy"};

        // Create new nominal attribute
        ArrayList<String> labelList = new ArrayList<>();
        for (String label : labels) labelList.add(label);
        Attribute categoryAttr = new Attribute("Life Ladder Category", labelList);

        // Create new dataset with added attribute
        ArrayList<Attribute> newAttrs = new ArrayList<>();
        for (int i = 0; i < data.numAttributes(); i++) newAttrs.add(data.attribute(i));
        newAttrs.add(categoryAttr);

        Instances newData = new Instances(data.relationName() + "_transformed", newAttrs, data.numInstances());

        for (int r = 0; r < data.numInstances(); r++) {
            Instance old = data.instance(r);
            double lifeVal = old.value(lifeLadder.index());
            String label = null;
            for (int i = 0; i < bins.length - 1; i++) {
                if (lifeVal >= bins[i] && lifeVal <= bins[i + 1]) {
                    label = labels[i];
                    break;
                }
            }

            Instance inst = new DenseInstance(newData.numAttributes());
            inst.setDataset(newData);
            for (int a = 0; a < data.numAttributes(); a++) {
                inst.setValue(a, old.value(a));
            }
            if (label != null) {
                inst.setValue(newData.numAttributes() - 1, label);
            } else {
                inst.setMissing(newData.numAttributes() - 1);
            }
            newData.add(inst);
        }

        System.out.println("Added Life Ladder Category attribute.");
        return newData;
    }

    public static Instances renameAndReorderLifeLadder(Instances data) throws Exception {
        Attribute lifeLadder = data.attribute("Life Ladder");
        Attribute category = data.attribute("Life Ladder Category");

        if (lifeLadder == null || category == null) {
            System.out.println("Missing Life Ladder or Life Ladder Category attribute.");
            return data;
        }

        // Step 1: Create new attribute list
        ArrayList<Attribute> reordered = new ArrayList<>();

        // Add all attributes except Life Ladder and Category
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute att = data.attribute(i);
            if (!att.name().equals("Life Ladder") && !att.name().equals("Life Ladder Category")) {
                reordered.add(att);
            }
        }

        // Step 2: Add renamed Life Ladder
        Attribute renamedLifeLadder = new Attribute("Life Ladder Numeric");
        reordered.add(renamedLifeLadder);

        // Step 3: Add Life Ladder Category at the end
        ArrayList<String> categoryLabels = new ArrayList<>();
        for (int i = 0; i < category.numValues(); i++) {
            categoryLabels.add(category.value(i));
        }
        Attribute newCategory = new Attribute("Life Ladder Category", categoryLabels);
        reordered.add(newCategory);

        // Step 4: Create new dataset
        Instances newData = new Instances(data.relationName() + "_reordered", reordered, data.numInstances());

        for (int r = 0; r < data.numInstances(); r++) {
            Instance old = data.instance(r);
            Instance inst = new DenseInstance(reordered.size());
            inst.setDataset(newData);

            for (int a = 0; a < reordered.size(); a++) {
                Attribute att = reordered.get(a);
                if (att.name().equals("Life Ladder Numeric")) {
                    inst.setValue(a, old.value(lifeLadder));
                } else if (att.name().equals("Life Ladder Category")) {
                    inst.setValue(a, old.stringValue(category));
                } else {
                    Attribute oldAtt = data.attribute(att.name());
                    if (oldAtt != null) {
                        if (old.isMissing(oldAtt)) {
                            inst.setMissing(a);
                        } else if (att.isNumeric()) {
                            inst.setValue(a, old.value(oldAtt));
                        } else {
                            inst.setValue(a, old.stringValue(oldAtt));
                        }
                    }
                }
            }

            newData.add(inst);
        }

        System.out.println("Renamed and reordered Life Ladder attributes.");
        return newData;
    }
}