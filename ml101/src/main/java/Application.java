import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Application
{
    /*
    Supervised and unsupervised learning are the most popular
    approaches to machinge learning.
    Both require feeding the machine a massive data records to correlate
    and learn from commonly called feature vectors.
    For exp in case of an individual house, a feature vector might
    consists of features such as overall house size, number of rooms,
    and the age of the house.

    Supervised ML:
    A machine is fed with feature vectors and associated labels.

    Unsupervised ML:
    Is programmed to predict answers without human labeling, or even
    questions. Rather than predtermine lables or what the results should be,
    unsupervised learning harness massive data sets and processing powet to discover
    previously unknown correlations.

    The challenge of supervised machine learning is to find the proper
    prediction function for a specific question.
    Given the concrete target function, the function can be used to make a prediction
    for each feature vector x.

    y = h(x)

    //target function h (which is the output of the learn process)
    Function <Double[], Double> h = ...;

    //set the feature vector with house size-101 and number of rooms-3
    Double[]x = new Double[]{101.0, 3.0};

    //and predicted the house price(label)
    double y = h.apply(x);

    Linear Regression:
    Assumes the relationship between input features and outputted label is linear.
    The generic linear regression function returns the predicted value by summarizing
    each elements of the feature vector multiplied by a theta parameter.
    The theta parameters are used within the training process to adapt or "tune" the
    regression function based on the training data.

    h(x) = Theta0 * 1 + Theta1 * x1 + ... + Thetan * xn
    Note that feature x0 is a constant offset term set with the value 1 for
    computational purposes. As a result, the index of a domain specific feature such as
    house size will start with x1.

    Scoring the target function:
    In ML, a cost function is used to compute the mean error or target function.

    Training the target function:
    You can use the gradient descent algorithm for computing the best-fitting
    theta parameters.

    Gradient Descent:
    minimizes the cost function, meaning that it's used to find the theta
    combinations that produces the lowest cost based on the training data.

    Underfitting is used to indicate that the learning algorithm does not caputure
    the underlying trend of the data.


    Adding Features and Feature Scaling:
    If you discover that your target function does not fit the problem you
    are trying to solve, you can adjust it. A common way to correct the underfitting
    is to add more features into the feature vector.
    Using multiple features requires feature scaling, which is used to standardize the
    range of different features.

    Overfitting and cross-validation:
    Overfitting occurs when the target function or model fits the training
    data too well, by capturing noise or random fluctuations in the training data.
    In cross-validation, you evaluate the trained models using an unseen validation data set
    after the learning process has completed. The available, labeled data set will be
    split into three parts:
    1. training data set
    2. validation data set
    3. test data set


     */


    public static void scalingExample()
    {

        // create the dataset
        List<Double[]> dataset = new ArrayList<>();
        dataset.add(new Double[] { 1.0,  90.0,  8100.0 });   // feature vector of house#1
        dataset.add(new Double[] { 1.0, 101.0, 10201.0 });   // feature vector of house#2
        dataset.add(new Double[] { 1.0, 103.0, 10609.0 });   // ...
        //...

        // create the labels
        List<Double> labels = new ArrayList<>();
        labels.add(249.0);        // price label of house#1
        labels.add(338.0);        // price label of house#2
        labels.add(304.0);        // ...
        //...

        // scale the extended feature list
        Function<Double[], Double[]> scalingFunc = FeaturesScaling.createFunction(dataset);
        List<Double[]>  scaledDataset  = dataset.stream().map(scalingFunc).collect(Collectors.toList());

        // create hypothesis function with initial thetas and train it with learning rate 0.1
        LinearRegressionFunction targetFunction =  new LinearRegressionFunction(new double[] { 1.0, 1.0, 1.0 });
        for (int i = 0; i < 10000; i++) {
            targetFunction = train(targetFunction, scaledDataset, labels, 0.1);
        }


    // make a prediction of a house with size if 600 m2
        Double[] scaledFeatureVector = scalingFunc.apply(new Double[] { 1.0, 600.0, 360000.0 });
        double predictedPrice = targetFunction.apply(scaledFeatureVector);
    }

    public static LinearRegressionFunction train(LinearRegressionFunction targetFunction,
                                                 List<Double[]> dataset,
                                                 List<Double> labels,
                                                 double alpha)
    {
        int m = dataset.size();
        double[] thetaVector = targetFunction.getThetas();
        double[] newThetaVector = new double[thetaVector.length];

        //compute the new theta of each element of the theta array
        for (int j = 0; j < thetaVector.length; j++)
        {
            //summarize the error gap * feature

            double sumErrors = 0 ;
            for (int i = 0; i< m; i ++)
            {
                Double[] featureVector = dataset.get(i);
                double error = targetFunction.apply(featureVector) - labels.get(i);
                sumErrors += error * thetaVector[j];
            }

            //compute the new theta value
            double gradient = (1.0/m) * sumErrors;
            newThetaVector[j] = thetaVector[j] - alpha * gradient;
        }
        return  new LinearRegressionFunction(newThetaVector);
    }

    public static double cost(Function<Double[], Double> targetFunction,
                              List<Double[]> dataset,
                              List<Double> labels)
    {
        int m = dataset.size();
        int sumSquaredErrors = 0;
        //calculate teh equared error ("gap") for each training example and add it to
        //total sum

        for (int i = 0; i < m; i++)
        {
            //get the feature vector of the current example
            Double [] featureVector = dataset.get(i);
            //predict the value and compute this error based on the real value (label)
           double predicted = targetFunction.apply(featureVector);
           double label = labels.get(i);
           double gap = predicted - label;
           sumSquaredErrors += Math.pow(gap, 2);
        }
        //calculate and retrun the mean value of the errors (the smaller the better)
        return (1.0 / (2 * m)) * sumSquaredErrors;

    }

    public double testLinearRegression()
    {
        //the theta vector used here was output of a train process
        double [] thetaVector = new double[] {1.004579, 5.286822};
        LinearRegressionFunction targetFunction = new LinearRegressionFunction(thetaVector);

        //create the feature vector function with x0 = 1 (for computational reasons and x1 = house size
        Double [] featureVector = new Double[] {1.0, 1330.0};

        // make the prediction
        return targetFunction.apply(featureVector);
    }


    public static void main(String [] args)
    {
        System.out.println(new Application().testLinearRegression());
    }


}
