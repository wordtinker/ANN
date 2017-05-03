using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.EA.Train;
using Encog.Neural.NEAT;
using Encog.Neural.Networks.Training;
using Encog.Util.Simple;
using System;
using System.IO;

namespace XORNeat
{
    class Program
    {
        /// <summary>
        /// Input for the XOR function.
        /// </summary>
        public static double[][] XORInput = {
                                                new double[2] {0.0, 0.0},
                                                new double[2] {1.0, 0.0},
                                                new double[2] {0.0, 1.0},
                                                new double[2] {1.0, 1.0}
                                            };

        /// <summary>
        /// Ideal output for the XOR function.
        /// </summary>
        public static double[][] XORIdeal = {
                                                new double[1] {0.0},
                                                new double[1] {1.0},
                                                new double[1] {1.0},
                                                new double[1] {0.0}
                                            };
        static void Main(string[] args)
        {
            // this form of ANN uses genetic algorithm to produce
            // hidden layer of neurons
            // A NEAT network starts with only an
            // input layer and output layer. The rest is evolved as the training progresses.
            // Connections inside of a NEAT neural network can be feedforward, recurrent,
            // or self - connected.All of these connection types will be tried by NEAT as it
            // attempts to evolve a neural network capable of the given task.
            IMLDataSet trainingSet = new BasicMLDataSet(XORInput, XORIdeal);
            NEATPopulation pop = new NEATPopulation(2, 1, 1000);
            pop.Reset();
            pop.InitialConnectionDensity = 1.0; // not required, but speeds processing.
            ICalculateScore score = new TrainingSetScore(trainingSet);
            // train the neural network
            TrainEA train = NEATUtil.ConstructNEATTrainer(pop, score);

            EncogUtility.TrainToError(train, 0.01);

            NEATNetwork network = (NEATNetwork)train.CODEC.Decode(train.BestGenome);
            // TODO no persistance? no means to peek structure?

            // test the neural network
            Console.WriteLine(@"Neural Network Results:");
            EncogUtility.Evaluate(network, trainingSet);

        }
    }
}
