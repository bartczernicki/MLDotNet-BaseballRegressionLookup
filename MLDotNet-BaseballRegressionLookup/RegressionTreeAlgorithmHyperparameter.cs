using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace MLDotNet_BaseballRegressionLookup
{
    public class RegressionTreeAlgorithmHyperparameter : AlgorithmHyperparameter
    {
        public int MinimumDataPointsInLeaves { get; set; }
        public double LearningRate { get; set; }
        public int NumberOfLeaves { get; set; }
        public int NumberOfTrees { get; set; }

        public RegressionMetrics RegressionMetrics { get; set; }

        public override string ToString()
        {
            return string.Format("MinimumDataPointsInLeaves: {0} LearningRate: {1} NumberOfLeaves: {2} NumberOfTrees: {3}",
                MinimumDataPointsInLeaves, LearningRate, NumberOfLeaves, NumberOfTrees);
        }
    }
}
