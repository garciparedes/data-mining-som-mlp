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

    private static final String OJOSECO_FILEPATH = "./digitos.entrena.normalizados.output.csv";
    private static final double RATIO_TEST = 0.66;

    public static void main(String[] args) {
        try {


            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(OJOSECO_FILEPATH));



            Instances data = loader.getDataSet();

            Normalize normalize = new Normalize();
            normalize.setInputFormat(data);
            data = Filter.useFilter(data, normalize);

            NumericToNominal convert= new NumericToNominal();
            convert.setInputFormat(data);
            convert.setAttributeIndices("last");

            data = Filter.useFilter(data, convert);
            data.setClassIndex(data.numAttributes() - 1);

            System.out.println(data.toSummaryString());



            data.randomize(new Random(0));

            int trainSize = Math.toIntExact(Math.round(data.numInstances() * RATIO_TEST));
            int testSize = data.numInstances() - trainSize;

            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);



            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setOptions(Utils.splitOptions("-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a"));
            mlp.buildClassifier(train);

            System.out.println(mlp.toString());



            Evaluation eval = new Evaluation(test);
            eval.evaluateModel(mlp, test);

            System.out.println(eval.toSummaryString());


        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
