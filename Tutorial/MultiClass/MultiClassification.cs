using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace ml.Tutorial.MultiClass
{
    class MultiClassification : MLTestTrain<GitHubIssues>
    {
        public MultiClassification() 
            : base("Многоклассовая", 
            "Tutorial/multiclass/issues_train.tsv", 
            "Tutorial/multiclass/issues_test.tsv",  0) {}

        protected override ITransformer BuildAndTrainModel()
        {
            var pipeline = context.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(context.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(context);

            var trainingPipeLine = pipeline.Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trainedPipeLine = trainingPipeLine.Fit(trainData);

            return trainedPipeLine;
        }

        protected override void Evaluate(ITransformer model)
        {
            var metrics = context.MulticlassClassification.Evaluate(model.Transform(testData));
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {metrics.MicroAccuracy:0.###} => 1");
            Console.WriteLine($"*       MacroAccuracy:    {metrics.MacroAccuracy:0.###} => 1");
            Console.WriteLine($"*       LogLoss:          {metrics.LogLoss:0.###} => 0");
            Console.WriteLine($"*       LogLossReduction: {metrics.LogLossReduction:0.###} => 0");
            Console.WriteLine($"*************************************************************************************************************");

        }

        protected override void UseModel(ITransformer model)
        {
            var predEngine = context.Model.CreatePredictionEngine<GitHubIssues, Prediction>(model);
            GitHubIssues issue = new GitHubIssues
            {
                Title = "Веб-сокеты работают слишком медленно!",
                Description = "WebSockets-соединение через SignalR выглядит слишком медленно работающим на моей машине.."
            };

            var prediction = predEngine.Predict(issue);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
        }
    }
}
