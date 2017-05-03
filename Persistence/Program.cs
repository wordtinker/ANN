using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Persist;
using Encog.Util.Simple;
using System;
using System.IO;

namespace Persistence
{
    class Program
    {
        public const String FILENAME = "encogexample.eg";
        /// <summary>
        /// Input for the XOR function.
        /// </summary>
        public static double[][] XOR_INPUT = {
                                                 new double[2] {0.0, 0.0},
                                                 new double[2] {1.0, 0.0},
                                                 new double[2] {0.0, 1.0},
                                                 new double[2] {1.0, 1.0}
                                             };

        /// <summary>
        /// Ideal output for the XOR function.
        /// </summary>
        public static double[][] XOR_IDEAL = {
                                                 new double[1] {0.0},
                                                 new double[1] {1.0},
                                                 new double[1] {1.0},
                                                 new double[1] {0.0}
                                             };
        static void Main(string[] args)
        {
            IMLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
            BasicNetwork network = EncogUtility.SimpleFeedForward(2, 6, 0, 1, false);
            EncogUtility.TrainToError(network, trainingSet, 0.01);
            double error = network.CalculateError(trainingSet);
            Console.WriteLine($"Error before save to EG: {error}");
            EncogDirectoryPersistence.SaveObject(new FileInfo(FILENAME), network);
            network = (BasicNetwork)EncogDirectoryPersistence.LoadObject(new FileInfo(FILENAME));
            error = network.CalculateError(trainingSet);
            Console.WriteLine($"Error after load from EG: {error}");
        }
    }
}
