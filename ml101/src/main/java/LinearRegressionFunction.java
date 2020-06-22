import java.util.Arrays;
import java.util.function.Function;

public class LinearRegressionFunction implements Function<Double[], Double>
{
    private final double[] thetaVector;

    LinearRegressionFunction(double[] thetaVector)
    {
        this.thetaVector = Arrays.copyOf(thetaVector, thetaVector.length);
    }
    @Override
    public Double apply(Double[] featureVector) {
        // For computational reasons the first element has to be 1.0
        assert featureVector[0] == 1.0;

        // simple, sequentail implementation
        double prediction = 0;
        for (int j = 0; j <thetaVector.length; j ++)
        {
            prediction += thetaVector[j] * featureVector[j];
        }
        return  prediction;
    }

    public double[] getThetas()
    {
        return Arrays.copyOf(thetaVector, thetaVector.length);
    }

}
