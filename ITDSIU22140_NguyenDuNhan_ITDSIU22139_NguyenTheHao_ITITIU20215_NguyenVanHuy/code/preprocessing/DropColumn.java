package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.preprocessing;

import weka.core.Instances;
import weka.core.Attribute;

import java.util.List;

/**
 * Utility to drop columns (attributes) by name.
 */
public class DropColumn {

    public static void dropColumns(Instances data, List<String> colNames) {
        // Delete from last to first to keep indexes valid
        for (String name : colNames) {
            Attribute att = data.attribute(name);
            if (att == null) {
                System.out.println("Warning: cannot drop, attribute not found: " + name);
                continue;
            }
            int idx = att.index();
            data.deleteAttributeAt(idx);
            System.out.println("Dropped attribute: " + name);
        }
    }
}
