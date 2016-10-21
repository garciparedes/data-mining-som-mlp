import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.util.Random;
import java.io.*;

/**
 * Created by garciparedes on 16/10/2016.
 */
public class WekaMultiLayerPerceptron {

    private static final String ENTRENA_FILEPATH = "./digitos.entrena.normalizados.output.csv";
    private static final String TEST_FILEPATH = "./digitos.test.normalizados.output.csv";


    public static void main(String[] args) {
        try {
            Instances train = getInstancesFromFile(ENTRENA_FILEPATH);
            Instances test = getInstancesFromFile(TEST_FILEPATH);

            //System.out.println(train.toSummaryString());
            //System.out.println(test.toSummaryString());


            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setTrainingTime(12000);
            mlp.setHiddenLayers("96");
            mlp.buildClassifier(train);
            //System.out.println(mlp.toString());


            Evaluation eval = new Evaluation(test);
            eval.evaluateModel(mlp, test);
            System.out.println(eval.toSummaryString());

        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    private static Instances getInstancesFromFile(String filePath) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(filePath));

        Instances data = loader.getDataSet();

        Normalize normalize = new Normalize();
        normalize.setInputFormat(data);
        data = Filter.useFilter(data, normalize);

        NumericToNominal convert= new NumericToNominal();
        convert.setInputFormat(data);
        convert.setAttributeIndices("last");

        data = Filter.useFilter(data, convert);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
}
