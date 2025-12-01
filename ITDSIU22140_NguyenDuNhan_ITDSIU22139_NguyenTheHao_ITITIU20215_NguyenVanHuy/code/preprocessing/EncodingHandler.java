package ITDSIU22140_NguyenDuNhan_ITDSIU22139_NguyenTheHao_ITITIU20215_NguyenVanHuy.code.preprocessing;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

/**
 * EN: Encoding utilities (one-hot encode nominal attributes).
 * VI: Các hàm mã hóa (one-hot) cho thuộc tính danh nghĩa.
 */
public class EncodingHandler {

    /**
     * One-hot encode a nominal attribute using Weka's NominalToBinary.
     * EN: Similar to pd.get_dummies(df[[attr]], ...).
     * VI: Tương đương get_dummies cho 1 cột nominal.
     */
    public static Instances oneHotEncodeNominal(Instances data, String attrName) throws Exception {
        Attribute att = data.attribute(attrName);
        if (att == null) {
            System.out.println("Warning: cannot one-hot, attribute not found: " + attrName);
            return data;
        }
        if (!att.isNominal()) {
            System.out.println("Warning: attribute not nominal for one-hot: " + attrName);
            return data;
        }

        int oneBasedIndex = att.index() + 1; // Weka filters use 1-based index strings
        String idxStr = String.valueOf(oneBasedIndex);

        NominalToBinary ntb = new NominalToBinary();
        ntb.setAttributeIndices(idxStr);     // only this attribute
        ntb.setBinaryAttributesNominal(false); // numeric 0/1

        ntb.setInputFormat(data);
        Instances newData = Filter.useFilter(data, ntb);

        System.out.println("One-hot encoded attribute: " + attrName);
        return newData;
    }
}
