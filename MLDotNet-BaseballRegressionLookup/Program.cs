using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Protobuf;
using Microsoft.Data.DataView; // Required for Dataview
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Normalizers;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Newtonsoft.Json;
using System.Diagnostics;

namespace MLDotNet_BaseballRegressionLookup
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "BaseballHOFTrainingv2.csv");
        private static string _validationDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "BaseballHOFValidationv2.csv");

        private static MLContext _mlContext;

        // Regression ML model(s) will try to predict Hits
        private static string _labelColunmn = "H";

        private static string[] featureColumns = new string[] {
            "AB", "BattingAverage" };

        private static string[] algorithmsForModelExplainability = new string[] {
                "FastTree", "FastTreeTweedie",
                "GeneralizedAdditiveModels", "OnlineGradientDescent",
                "PoissonRegression", "StochasticDualCoordinateAscent"};

        private static int mlContextSeed = 100;

        static void Main(string[] args)
        {
            Console.WriteLine("Starting Training Job.");
            Console.WriteLine();

            #region Step 1) ML.NET Setup & Load Data

            Console.WriteLine("##########################");
            Console.WriteLine("Step 1: Load Data...");
            Console.WriteLine("##########################\n");

            // Set the seed explicitly for reproducability (models will be built with consistent results)
            _mlContext = new MLContext(seed: mlContextSeed);

            // Read the training/validation data from a text file
            var dataTrain = _mlContext.Data.ReadFromTextFile<MLBBaseballBatter>(path: _trainDataPath,
                hasHeader: true, separatorChar: ',', allowQuotedStrings: false);
            var dataValidation = _mlContext.Data.ReadFromTextFile<MLBBaseballBatter>(path: _validationDataPath,
                hasHeader: true, separatorChar: ',', allowQuotedStrings: false);

            #if DEBUG
            // Debug Only: Preview the training/validation data
            var dataTrainPreview = dataTrain.Preview();
            var dataValidationPreview = dataValidation.Preview();
            #endif

            // Cache the loaded data
            var cachedTrainData = _mlContext.Data.Cache(dataTrain);
            var cachedValidationData = _mlContext.Data.Cache(dataValidation);

            #endregion

            #region Step 2) Build Multiple Machine Learning Models

            // Notes:
            // Model training is for demo purposes and uses the default hyperparameters.
            // Default parameters were used in optimizing for large data sets.
            // It is best practice to always provide hyperparameters explicitly in order to have historical reproducability
            // as the ML.NET API evolves.

            Console.WriteLine("##########################");
            Console.WriteLine("Step 2: Train Models...");
            Console.WriteLine("##########################\n");

            /* FAST TREE MODELS */
            Console.WriteLine("Training...Fast Tree Models.");

            // Build simple data pipeline
            var learningPipelineFastTreeHits =
                Utilities.GetBaseLinePipeline(_mlContext, featureColumns).Append(
                _mlContext.Regression.Trainers.FastTree(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelPipelineFastTreeHits = learningPipelineFastTreeHits.Fit(cachedTrainData);
            // Save the model to storage
            Utilities.SaveModel(_appPath, _mlContext, "FastTree", _labelColunmn, modelPipelineFastTreeHits);
            Utilities.SaveOnnxModel(_appPath, "FastTree", _labelColunmn, modelPipelineFastTreeHits, _mlContext, cachedTrainData);

            /* FAST TREE TWEEDIE MODELS */
            Console.WriteLine("Training...Fast Tree Tweedie Models.");

            // Build simple data pipeline
            var learningPipelineFastTreeTweedieHits =
                Utilities.GetBaseLinePipeline(_mlContext, featureColumns).Append(
                _mlContext.Regression.Trainers.FastTreeTweedie(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelPipelineFastTreeTweedieHits = learningPipelineFastTreeTweedieHits.Fit(cachedTrainData);
            // Save the model to storage
            Utilities.SaveModel(_appPath, _mlContext, "FastTreeTweedie", _labelColunmn, modelPipelineFastTreeTweedieHits);
            Utilities.SaveOnnxModel(_appPath, "FastTreeTweedie", _labelColunmn, modelPipelineFastTreeTweedieHits, _mlContext, cachedTrainData);

            /* GENERALIZED ADDITIVE MODELS */
            Console.WriteLine("Training...Generalized Additive Models.");

            // Build simple data pipeline
            var learningPipelineGeneralizedAdditiveModelsHits =
                Utilities.GetBaseLinePipeline(_mlContext, featureColumns).Append(
                _mlContext.Regression.Trainers.GeneralizedAdditiveModels(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelPipelineGeneralizedAdditiveModelsHits = learningPipelineGeneralizedAdditiveModelsHits.Fit(cachedTrainData);
            // Save the model to storage
            Utilities.SaveModel(_appPath, _mlContext, "GeneralizedAdditiveModels", _labelColunmn, modelPipelineGeneralizedAdditiveModelsHits);
            Utilities.SaveOnnxModel(_appPath, "GeneralizedAdditiveModels", _labelColunmn, modelPipelineGeneralizedAdditiveModelsHits, _mlContext, cachedTrainData);

            /* ONLINE GRADIENT DESCENT MODELS */
            Console.WriteLine("Training...Online Gradient Descent Models.");

            // Build simple data pipeline
            var learningPipelineOnlineGradientDescentHits =
                Utilities.GetBaseLinePipeline(_mlContext, featureColumns).Append(
                _mlContext.Regression.Trainers.OnlineGradientDescent(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelPipelineOnlineGradientDescentHits = learningPipelineOnlineGradientDescentHits.Fit(cachedTrainData);
            // Save the model to storage
            Utilities.SaveModel(_appPath, _mlContext, "OnlineGradientDescent", _labelColunmn, modelPipelineOnlineGradientDescentHits);
            Utilities.SaveOnnxModel(_appPath, "OnlineGradientDescent", _labelColunmn, modelPipelineOnlineGradientDescentHits, _mlContext, cachedTrainData);


            /* POISSON REGRESSION MODELS */
            Console.WriteLine("Training...Poisson Regression Models.");

            // Build simple data pipeline
            var learningPipelinePoissonRegressionHits =
                Utilities.GetBaseLinePipeline(_mlContext, featureColumns).Append(
                _mlContext.Regression.Trainers.PoissonRegression(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelPoissonRegressionHits = learningPipelinePoissonRegressionHits.Fit(cachedTrainData);
            // Save the model to storage
            Utilities.SaveModel(_appPath, _mlContext, "PoissonRegression", _labelColunmn, modelPoissonRegressionHits);
            Utilities.SaveOnnxModel(_appPath, "PoissonRegression", _labelColunmn, modelPoissonRegressionHits, _mlContext, cachedTrainData);

            /* STOCHASTIC DUAL COORDINATE ASCENT MODELS */
            Console.WriteLine("Training...Stochastic Dual Coordinate Ascent Models.");

            // Build simple data pipeline
            var learningPipelineStochasticDualCoordinateAscentHits =
                Utilities.GetBaseLinePipeline(_mlContext, featureColumns).Append(
                _mlContext.Regression.Trainers.StochasticDualCoordinateAscent(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelStochasticDualCoordinateAscentHits = learningPipelineStochasticDualCoordinateAscentHits.Fit(cachedTrainData);
            // Save the model to storage
            Utilities.SaveModel(_appPath, _mlContext, "StochasticDualCoordinateAscent", _labelColunmn, modelStochasticDualCoordinateAscentHits);
            Utilities.SaveOnnxModel(_appPath, "StochasticDualCoordinateAscent", _labelColunmn, modelStochasticDualCoordinateAscentHits, _mlContext, cachedTrainData);

            Console.WriteLine();

            #endregion

            #region Step 3) Report Performance Metrics

            Console.WriteLine("##########################");
            Console.WriteLine("Step 3: Report Metrics...");
            Console.WriteLine("##########################\n");

            for (int i = 0; i < algorithmsForModelExplainability.Length; i++)
            {
                var regressionMetrics = Utilities.GetRegressionModelMetrics(_appPath, _mlContext, _labelColunmn, algorithmsForModelExplainability[i], cachedValidationData);

                Console.WriteLine("Evaluation Metrics for " + algorithmsForModelExplainability[i] + " | " + _labelColunmn);
                Console.WriteLine("******************");
                Console.WriteLine("L1:         " + Math.Round(regressionMetrics.L1, 5).ToString());
                Console.WriteLine("L2:         " + Math.Round(regressionMetrics.L2, 5).ToString());
                Console.WriteLine("LossFn:     " + Math.Round(regressionMetrics.LossFn, 5).ToString());
                Console.WriteLine("Rms:        " + Math.Round(regressionMetrics.Rms, 5).ToString());
                Console.WriteLine("RSquared:   " + Math.Round(regressionMetrics.RSquared, 5).ToString());
                Console.WriteLine("******************");

                var loadedModel = Utilities.LoadModel(_mlContext, Utilities.GetModelPath(_appPath, algorithmName: algorithmsForModelExplainability[i], isOnnx: false, label: _labelColunmn));
                var transformedModelData = loadedModel.Transform(cachedValidationData);
                TransformerChain<ITransformer> lastTran = (TransformerChain<ITransformer>) loadedModel.LastTransformer;
                var enumerator = lastTran.GetEnumerator();

                Console.WriteLine("******************");
                Console.WriteLine();
            }

            Console.WriteLine();
            #endregion

            #region Step 4) Build Models Using Random Hyperparameter Search

            Console.WriteLine("##########################");
            Console.WriteLine("Step 4: Hyperparameter Random Search...");
            Console.WriteLine("##########################\n");


            var hyperparameterPerformanceMetricResults = new ConcurrentBag<RegressionTreeAlgorithmHyperparameter>();

            var dateTime = DateTime.Now;

            // Build a list of Random Search Hyperparameters
            var maxNumberOfIterations = 20;
            var numberOfIterationsArray = Enumerable.Range(0, maxNumberOfIterations).ToArray();
            ParallelOptions options = new ParallelOptions();
            options.MaxDegreeOfParallelism = 14;  // usually set this to number of available worker threads

            Parallel.For(0, numberOfIterationsArray.Length, options,
                         i =>
                         {
                             var algorithmName = "FastTree";
                             var mlContextSeed = 100;
                             var iteration = i + 1;
                             var labelColumn = "H";
                             var appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

                             var minLearningRate = 0.005;
                             var maxLearningRate = 2;

                             // Generate hyperparameters based on random values in ranges
                             // Note: In production, you would use MBO or more advanced statistical distributions
                             // Currently doing a pseudo-uniform distribution over pragmatic ranges
                             var newRandom = new Random((int)DateTime.Now.Ticks);
                             var numberOfleaves = newRandom.Next(2, 1200); // minimum is 2
                             var numberOfTrees = newRandom.Next(1, 1200);
                             var minDataPointsInTrees = newRandom.Next(1, 25);
                             var learningRate = newRandom.NextDouble() * (maxLearningRate - minLearningRate) + minLearningRate;
                             var modelName = algorithmName + "RandomSearch" + iteration;

                             // Add the metrics to the Concurrent Bag (List)
                             var regressionTreeHyperParameters =
                                                              new RegressionTreeAlgorithmHyperparameter
                                                              {
                                                                  AlgorithmName = algorithmName,
                                                                  LabelColumn = labelColumn,
                                                                  Iteration = iteration,
                                                                  MLContextSeed = mlContextSeed,
                                                                  // Regression parameters
                                                                  LearningRate = learningRate,
                                                                  MinimumDataPointsInLeaves = minDataPointsInTrees,
                                                                  NumberOfLeaves = numberOfleaves,
                                                                  NumberOfTrees = numberOfTrees
                                                              };
                             hyperparameterPerformanceMetricResults.Add(regressionTreeHyperParameters);

                         }
                        );

            // Build a list of Tasks to parallelize the model creation using random hyperparameter search
            List<Task> modelTasks = new List<Task>();
            foreach(var modelTaskHyperParameters in hyperparameterPerformanceMetricResults)
            {
                var appPath = _appPath;
                var task = new Task<RegressionTreeAlgorithmHyperparameter>(
                    () =>
                ProcessModel(modelTaskHyperParameters, cachedValidationData, appPath)//;
                    );
                modelTasks.Add(task);

                task.Start();
            }

            // Wait for all of the tasks to complete
            Task.WaitAll(modelTasks.ToArray());

            // TODO: Remove
            //    Parallel.For(0, numberOfIterationsArray.Length, options,
            //i => {
            //    var algorithmName = "FastTree";
            //    var mlContextSeed = 300;
            //    var mlContext = new MLContext(seed: mlContextSeed, conc: 1);
            //    var iteration = i + 1;
            //    var labelColumn = "H";
            //    var appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

            //    var minLearningRate = 0.005;
            //    var maxLearningRate = 2;

            //    // Generate hyperparameters based on random values in ranges
            //    // Note: In production, you would use MBO or more advanced statistical distributions
            //    // Currently doing a pseudo-uniform distribution over pragmatic ranges
            //    var newRandom = new Random((int) DateTime.Now.Ticks);
            //    var numberOfleaves = newRandom.Next(2, 1200); // minimum is 2
            //    var numberOfTrees = newRandom.Next(1, 1200);
            //    var minDataPointsInTrees = newRandom.Next(1, 25);
            //    var learningRate = newRandom.NextDouble() * (maxLearningRate - minLearningRate) + minLearningRate;
            //    var modelName = algorithmName + "RandomSearch" + iteration;

            //    var learningPipelineFastTreeTweedieGridSearchHits =
            //       Utilities.GetBaseLinePipeline(mlContext, featureColumns).Append(
            //       mlContext.Regression.Trainers.FastTree(labelColumn: labelColumn, learningRate: learningRate,
            //       numLeaves: numberOfleaves, numTrees: numberOfTrees, minDatapointsInLeaves: minDataPointsInTrees)
            //       );
            //    // Fit (build a Machine Learning Model)
            //    var modelFastTreeTweedieGridSearchHits = learningPipelineFastTreeTweedieGridSearchHits.Fit(cachedTrainData);
            //    // Save the model to storage
            //    Utilities.SaveModel(_appPath, mlContext, modelName, labelColumn, modelFastTreeTweedieGridSearchHits);

            //    var transformedData = modelFastTreeTweedieGridSearchHits.Transform(cachedValidationData);

            //    // Evaluate the model
            //    var regressionMetrics = mlContext.Regression.Evaluate(transformedData, "H");

            //    // Add the metrics to the Concurrent Bag (List)
            //    var regressionTreeHyperParameters =
            //                                     new RegressionTreeAlgorithmHyperparameter
            //                                     {
            //                                         AlgorithmName = algorithmName,
            //                                         Iteration = iteration,
            //                                         MLContextSeed = mlContextSeed,
            //                                         LearningRate = learningRate,
            //                                         // Regression parameters
            //                                         MinimumDataPointsInLeaves = minDataPointsInTrees,
            //                                         NumberOfLeaves = numberOfleaves,
            //                                         NumberOfTrees = numberOfTrees,
            //                                         RegressionMetrics = regressionMetrics
            //                                     };
            //    hyperparameterPerformanceMetricResults.Add(regressionTreeHyperParameters);


            //    var consoleOutput = string.Format(
            //       "Evaluation Metrics for {0} | {1}\n" +
            //       "{2}\n" +
            //       "**********************************************\n" +
            //       "L1:         {3}\n" +
            //       "L2:         {4}\n" +
            //       "LossFn:     {5}\n" +
            //       "Rms:        {6}\n" +
            //       "RSquared    {7}\n",
            //        modelName, labelColumn,
            //        regressionTreeHyperParameters.ToString(),
            //        Math.Round(regressionMetrics.L1, 5).ToString(),
            //        Math.Round(regressionMetrics.L2, 5).ToString(),
            //        Math.Round(regressionMetrics.LossFn, 5).ToString(),
            //        Math.Round(regressionMetrics.Rms, 5).ToString(),
            //        Math.Round(regressionMetrics.RSquared, 5).ToString()
            //        );

            //    Console.Out.WriteLineAsync(consoleOutput);
            //});

            var elapsedSeconds = (DateTime.Now - dateTime).TotalSeconds;
            Console.WriteLine("Hyperparameter Random Search Time: " + elapsedSeconds + " seconds");
            Console.WriteLine();

            // Order by the best performing model, serialize to JSON
            var hyperparameterPerformanceMetricResultsOrdered = hyperparameterPerformanceMetricResults
                .OrderByDescending(a => a.RegressionMetrics.RSquared).ToList();
            var json = JsonConvert.SerializeObject(hyperparameterPerformanceMetricResultsOrdered, Formatting.Indented);
            var jsonPath = Path.Combine(_appPath, "..", "..", "..", "Models", "RandomSearchHyperParameters.json");
            File.WriteAllText(jsonPath, json);


            #endregion

            #region Step 5) New Predictions - Using Ficticious Player Data

            Console.WriteLine("##########################");
            Console.WriteLine("Step 5: New Predictions...");
            Console.WriteLine("##########################\n");

            // Set algorithm type to use for predictions
            var algorithmTypeName = "FastTree";
            var loadedModelHits = Utilities.LoadModel(_mlContext, (Utilities.GetModelPath(_appPath, algorithmTypeName, false, "H")));
            // Use the "best" model from the random search
            var loadedModelRandomSearchHits = Utilities.LoadModel(_mlContext, (Utilities.GetModelPath(_appPath,
                hyperparameterPerformanceMetricResultsOrdered[0].GetModelName(), false, "H")));

            //var transformedData = loadedModelHits.Transform(cachedValidationData);
            //var transformedData2 = loadedModelRandomSearchHits.Transform(cachedValidationData);
            //var test1 = transformedData.Preview(1000).RowView.ToArray();
            //var test2 = transformedData2.Preview(1000).RowView.ToArray();
            // Evaluate the model
            //var regressionMetricsTest = _mlContext.Regression.Evaluate(transformedData, "H");
            //var regressionMetricsTest2 = _mlContext.Regression.Evaluate(transformedData2, "H");

            // Create prediction engine
            var predEngineHits = loadedModelHits.CreatePredictionEngine<MLBBaseballBatter, HitsPredictions>(_mlContext);
            var predEngineRandomSearchHits = loadedModelRandomSearchHits.CreatePredictionEngine<MLBBaseballBatter, HitsPredictions>(_mlContext);

            // Create statistics for bad, average & great player
            var badMLBBatter = new MLBBaseballBatter
            {
                FullPlayerName = "Bad Player",
                ID = 100f,
                InductedToHallOfFame = false,
                LastYearPlayed = 0f,
                OnHallOfFameBallot = false,
                YearsPlayed = 2f,
                AB = 100f,
                R = 10f,
                H = 30f,
                Doubles = 1f,
                Triples = 1f,
                HR = 1f,
                RBI = 10f,
                SB = 10f,
                BattingAverage = 0.3f,
                SluggingPct = 0.15f,
                AllStarAppearances = 1f,
                MVPs = 0f,
                TripleCrowns = 0f,
                GoldGloves = 0f,
                MajorLeaguePlayerOfTheYearAwards = 0f,
                TB = 200f
            };
            var averageMLBBatter = new MLBBaseballBatter
            {
                FullPlayerName = "Average Player",
                ID = 100f,
                InductedToHallOfFame = false,
                LastYearPlayed = 0f,
                OnHallOfFameBallot = false,
                YearsPlayed = 2f,
                AB = 8393f,
                R = 1162f,
                H = 2340f,
                Doubles = 410f,
                Triples = 8f,
                HR = 439f,
                RBI = 1412f,
                SB = 9f,
                BattingAverage = 0.279f,
                SluggingPct = 0.486f,
                AllStarAppearances = 6f,
                MVPs = 0f,
                TripleCrowns = 0f,
                GoldGloves = 0f,
                MajorLeaguePlayerOfTheYearAwards = 0f,
                TB = 4083f
            };
            var greatMLBBatter = new MLBBaseballBatter
            {
                FullPlayerName = "Great Player",
                ID = 100f,
                InductedToHallOfFame = false,
                LastYearPlayed = 0f,
                OnHallOfFameBallot = false,
                YearsPlayed = 20f,
                AB = 10000f,
                R = 1900f,
                H = 3500f,
                Doubles = 500f,
                Triples = 150f,
                HR = 600f,
                RBI = 1800f,
                SB = 400f,
                BattingAverage = 0.350f,
                SluggingPct = 0.65f,
                AllStarAppearances = 14f,
                MVPs = 2f,
                TripleCrowns = 1f,
                GoldGloves = 4f,
                MajorLeaguePlayerOfTheYearAwards = 2f,
                TB = 7000f
            };

            var batters = new List<MLBBaseballBatter> { badMLBBatter, averageMLBBatter, greatMLBBatter };
            // Convert the list to an IDataView
            var newPredictionsData = _mlContext.Data.ReadFromEnumerable(batters);

            // Make the predictions for both OnHallOfFameBallot & InductedToHallOfFame
            var predBadHits = predEngineHits.Predict(badMLBBatter);
            var predAverageHits = predEngineHits.Predict(averageMLBBatter);
            var predGreatHits = predEngineHits.Predict(greatMLBBatter);
            var predBadRandomSearchHits = predEngineRandomSearchHits.Predict(badMLBBatter);
            var predAverageRandomSearchHits = predEngineRandomSearchHits.Predict(averageMLBBatter);
            var predGreatRandomSearchHits = predEngineRandomSearchHits.Predict(greatMLBBatter);

            // Report the results
            Console.WriteLine("Algorithm Used for Model Prediction: " + algorithmTypeName);
            Console.WriteLine("Bad Baseball Player Prediction");
            Console.WriteLine("------------------------------");
            Console.WriteLine("Hits Prediction: " + predBadHits.Hits.ToString() + " | " + "Actual Hits: " + badMLBBatter.H);
            Console.WriteLine("Hits Prediction: " + predBadRandomSearchHits.Hits.ToString() + " | " + "Actual Hits: " + badMLBBatter.H + " (Best Random Search Model)");
            Console.WriteLine();
            Console.WriteLine("Average Baseball Player Prediction");
            Console.WriteLine("------------------------------");
            Console.WriteLine("Hits Prediction: " + predAverageHits.Hits.ToString() + " | " + "Actual Hits: " + averageMLBBatter.H);
            Console.WriteLine("Hits Prediction: " + predAverageRandomSearchHits.Hits.ToString() + " | " + "Actual Hits: " + averageMLBBatter.H + " (Best Random Search Model)");
            Console.WriteLine();
            Console.WriteLine("Great Baseball Player Prediction");
            Console.WriteLine("------------------------------");
            Console.WriteLine("Hits Prediction: " + predGreatHits.Hits.ToString() + " | " + "Actual Hits: " + greatMLBBatter.H);
            Console.WriteLine("Hits Prediction: " + predGreatRandomSearchHits.Hits.ToString() + " | " + "Actual Hits: " + greatMLBBatter.H + " (Best Random Search Model)");
            Console.WriteLine();

            #endregion

            Console.ReadLine();
        }

        public static RegressionTreeAlgorithmHyperparameter ProcessModel(RegressionTreeAlgorithmHyperparameter hyperParameters,
            IDataView validationData, string appPath//,
            //CancellationToken ct
            )
        {
            var mlContext = new MLContext(seed: hyperParameters.MLContextSeed);

            var learningPipelineGridSearchHits =
               Utilities.GetBaseLinePipeline(mlContext, featureColumns).Append(
               mlContext.Regression.Trainers.FastTree(labelColumn: hyperParameters.LabelColumn, learningRate: hyperParameters.LearningRate,
               numLeaves: hyperParameters.NumberOfLeaves, numTrees: hyperParameters.NumberOfTrees, minDatapointsInLeaves: hyperParameters.MinimumDataPointsInLeaves)
               );
            // Fit (build a Machine Learning Model)
            var modelGridSearchHits = learningPipelineGridSearchHits.Fit(validationData);
            // Save the model to storage
            Utilities.SaveModel(appPath, mlContext, hyperParameters.GetModelName(), hyperParameters.LabelColumn, modelGridSearchHits);
            // Utilities.SaveOnnxModel(appPath, hyperParameters.GetModelName(), hyperParameters.LabelColumn, modelGridSearchHits, mlContext, validationData);

            var transformedData = modelGridSearchHits.Transform(validationData);

            // Evaluate the model
            var regressionMetrics = mlContext.Regression.Evaluate(transformedData, hyperParameters.LabelColumn);
            hyperParameters.RegressionMetrics = regressionMetrics;

            var consoleOutput = string.Format(
               "Evaluation Metrics for {0} | {1}\n" +
               "{2}\n" +
               "**********************************************\n" +
               "L1:         {3}\n" +
               "L2:         {4}\n" +
               "LossFn:     {5}\n" +
               "Rms:        {6}\n" +
               "RSquared    {7}\n",
                hyperParameters.GetModelName(), hyperParameters.LabelColumn,
                hyperParameters.ToString(),
                Math.Round(regressionMetrics.L1, 5).ToString(),
                Math.Round(regressionMetrics.L2, 5).ToString(),
                Math.Round(regressionMetrics.LossFn, 5).ToString(),
                Math.Round(regressionMetrics.Rms, 5).ToString(),
                Math.Round(regressionMetrics.RSquared, 5).ToString()
                );

            Console.Out.WriteLineAsync(consoleOutput);

            return hyperParameters;
        }
    }
}
