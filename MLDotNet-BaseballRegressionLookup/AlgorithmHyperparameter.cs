using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLDotNet_BaseballRegressionLookup
{
    public class AlgorithmHyperparameter
    {
        public string AlgorithmName { get; set; }
        public int Iteration { get; set; }

        public int MinimumDataPointsInTrees { get; set; }
        public double LearningRate { get; set; }
        public int NumberOfLeaves { get; set; }
        public int NumberOfTrees { get; set; }

        public RegressionMetrics RegressionMetrics { get; set; }
    }
}
