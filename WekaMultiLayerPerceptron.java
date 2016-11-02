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


    private static final String ENTRENA_SOM_FILEPATH = "./digitos.entrena.normalizados.output.csv";
    private static final String TEST_SOM_FILEPATH = "./digitos.test.normalizados.output.csv";


    private static final String ENTRENA_FILEPATH = "./digitos.entrena.normalizados.input.csv";
    private static final String TEST_FILEPATH = "./digitos.test.normalizados.input.csv";



    public static void main(String[] args) {
        System.out.println();
        System.out.println("WEKA");
        System.out.println();


        System.out.println("SOM + MLP");
        runMLP(ENTRENA_SOM_FILEPATH, TEST_SOM_FILEPATH);

        System.out.println("MLP");
        runMLP(ENTRENA_FILEPATH, TEST_FILEPATH);
    }



    private static void runMLP(String trainFilePath, String testFilePath) {
        try {
            Instances train = getInstancesFromFile(trainFilePath);
            Instances test = getInstancesFromFile(testFilePath);

            //System.out.println(train.toSummaryString());
            //System.out.println(test.toSummaryString());


            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setTrainingTime(2000);
            mlp.setHiddenLayers("o");
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

        NumericToNominal convert= new NumericToNominal();
        convert.setInputFormat(data);
        convert.setAttributeIndices("last");

        data = Filter.useFilter(data, convert);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
}
