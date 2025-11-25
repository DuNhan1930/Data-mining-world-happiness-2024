import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;

public class TestWeka {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("data/weather.nominal.arff");
        Instances data = source.getDataSet();

        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        J48 tree = new J48();           // Decision tree
        tree.buildClassifier(data);

        System.out.println(tree);       // Print tree model
    }
}
