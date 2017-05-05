using Encog.Engine.Network.Activation;
using Encog.ML;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Train;
using Encog.ML.Train.Strategy;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Neural.Pattern;
using System;

namespace ElmanSin
{
    class Program
    {
        static double DegreeToRad(int degree)
        {
            return Math.PI * degree / 180.0;
        }

        static void Main(string[] args)
        {
            // used for prediction of time series
            // sin(x) in theory
            int DEGREES = 360;
            int WINDOW_SIZE = 16;
            double[][] Input = new double[DEGREES][];
            double[][] Ideal = new double[DEGREES][];

            // Create array of sin signals
            for (int i = 0; i < DEGREES; i++)
            {
                Input[i] = new double[WINDOW_SIZE];
                Ideal[i] = new double[] { Math.Sin(DegreeToRad(i + WINDOW_SIZE))};
                for (int j = 0; j < WINDOW_SIZE; j++)
                {
                    Input[i][j] = Math.Sin(DegreeToRad(i + j));
                }
            }
            // construct training set
            IMLDataSet trainingSet = new BasicMLDataSet(Input, Ideal);

            // construct an Elman type network
            // simple recurrent network
            ElmanPattern pattern = new ElmanPattern
            {
                InputNeurons = WINDOW_SIZE,
                ActivationFunction = new ActivationSigmoid(),
                OutputNeurons = 1
            };
            pattern.AddHiddenLayer(WINDOW_SIZE);
            IMLMethod method = pattern.Generate();
            BasicNetwork network = (BasicNetwork)method;
            // Train network
            IMLTrain train = new Backpropagation(network, trainingSet);
            var stop = new StopTrainingStrategy();
            train.AddStrategy(new Greedy());
            train.AddStrategy(stop);
            int epoch = 0;
            while (!stop.ShouldStop())
            {
                train.Iteration();
                Console.WriteLine($"Training Epoch #{epoch} Error:{train.Error}");
                epoch++;
            }
            // Test network
            foreach (IMLDataPair pair in trainingSet)
            {
                IMLData output = network.Compute(pair.Input);
                Console.WriteLine($"actual={output[0]}, ideal={pair.Ideal[0]}");
            }
        }
    }
}
