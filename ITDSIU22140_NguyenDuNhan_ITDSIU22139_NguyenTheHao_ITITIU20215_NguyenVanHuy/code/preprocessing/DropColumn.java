package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.preprocessing;

import weka.core.Instances;
import weka.core.Attribute;

import java.util.List;

/**
 * EN: Utility to drop columns (attributes) by name.
 * VI: Tiện ích để xóa cột theo tên thuộc tính.
 */
public class DropColumn {

    public static void dropColumns(Instances data, List<String> colNames) {
        // EN: Delete from last to first to keep indexes valid
        // VI: Xóa từ index lớn xuống nhỏ để không bị lệch index
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
