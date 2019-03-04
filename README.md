# MLDotNet-BaseballRegressionLookup

A .Net Core model building job that builds several models using MLB Baseball data from 1876 - 2017.  

The outcome is a single supervised learning regression prediction:
* Hits - based on AB (at bats) & BA (batting average), can machine learning be used to create an overfit model and basically build a lookup function for Hits.

Note: This model job is meant to be used as an exercise of "lookup functions" not necessarily

The model building job includes the following features:
* Builds multiple ML.NET regression models in a single C# "script" (job)
* Persists algorithm model hyperparameters in a JSON file
* Uses TPL (Task Parallel Library) to build 10s of models using multiple parallel threads
* Reports various performance metrics using a pre-defined holdout set
* Applies simple perscriptive/rules engine to select the "best model"
* Selected "best model" is used for inference on new ficticious baseball player careers (to verify closeness to Hits prediction)
* Persists the trained models in two different formats: native ML.NET and ONNX
* Loads the persisted models from storage and performs model explainability

Requirements:
* Visual Studio 2017, .NET Core, ML.NET v.10+
