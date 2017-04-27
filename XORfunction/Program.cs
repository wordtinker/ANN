using Encog.Engine.Network.Activation; // from NuGet
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Train;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using System;

namespace XORfunction
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1 create network

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
            // train
            IMLTrain train = new ResilientPropagation(network, trainingSet);
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
