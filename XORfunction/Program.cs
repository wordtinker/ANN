using Encog.Engine.Network.Activation; // from NuGet
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Train;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Neural.Networks.Training.Propagation.Manhattan;
using Encog.Neural.Networks.Training.Propagation.Quick;
using Encog.Neural.Networks.Training.Propagation.SCG;
using Encog.Neural.Networks.Training.Lma;
using System;

namespace XORfunction
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1 create feedforward network

            var network = new BasicNetwork();
            // add input layer with no activation function, one bias neuron and 2 inputs
            network.AddLayer(new BasicLayer(null, true, 2));
            // add hidden layer with Sigmoid activation function, one bias neuron and 3 inputs
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
            // add output layer
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
            network.Structure.FinalizeStructure();
            // init network to random weights
            network.Reset();

            // 2 train network

            double[][] XORInput = {
                new [ ] { 0.0 , 0.0 },
                new [ ] { 1.0 , 0.0 },
                new [ ] { 0.0 , 1.0 },
                new [ ] { 1.0 , 1.0 }
            };
            double[][] XORIdeal = {
                new [ ] { 0.0 },
                new [ ] { 1.0 },
                new [ ] { 1.0 },
                new [ ] {0.0}
            };
            // create training set
            IMLDataSet trainingSet = new BasicMLDataSet(XORInput, XORIdeal);
            // train using some form of backpropagation
            // backpropagation is best used if we have input and output sets
            // one of the best is resilient
            IMLTrain train = new ResilientPropagation(network, trainingSet);
            // other forms of propagation could be used
            //IMLTrain train = new Backpropagation(network, trainingSet, 0.7, 0.3);
            //IMLTrain train = new ManhattanPropagation(network, trainingSet, 0.00001);
            //IMLTrain train = new QuickPropagation(network, trainingSet, 2.0);
            //IMLTrain train = new ScaledConjugateGradient(network, trainingSet);
            // and non propagation, sometimes could be better than resilient
            //IMLTrain train = new LevenbergMarquardtTraining(network, trainingSet);

            int epoch = 1;
            do
            {
                train.Iteration();
                Console.WriteLine($"Epoch # {epoch} Error: {train.Error}");
                epoch++;
            } while (train.Error > 0.01);

            // 3 test the network
            Console.WriteLine("Neural Network Results:");
            foreach (IMLDataPair pair in trainingSet)
            {
                IMLData output = network.Compute(pair.Input);
                Console.WriteLine($"{pair.Input[0]}, {pair.Input[1]}, actual={output[0]}, ideal={pair.Ideal[0]}");
            }
        }
    }
}
