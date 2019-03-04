using System;
using System.Collections.Generic;
using System.Text;

namespace MLDotNet_BaseballRegressionLookup
{
    public class AlgorithmHyperparameter
    {
        public string AlgorithmName { get; set; }
        public int Iteration { get; set; }
        public int MLContextSeed { get; set; }
        public string LabelColumn { get; set; }

        public string GetModelName()
        {
            return string.Format("{0}-{1}", AlgorithmName, Iteration);
        }
    }
}
