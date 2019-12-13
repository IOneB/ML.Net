using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace ml.Tutorial.Binary
{
    internal class BinaryClassification : MLExample
    {
        const string _dataPath = "Tutorial/binary/yelp_labelled.txt";
        const string _modelPath = "model.zip";

        public BinaryClassification() : base("Двоичная классификация") {}

        protected override void LoadData()
        {
            var data = context.Data.LoadFromTextFile<SentimentialData>(_dataPath, hasHeader: false);
            var set = context.Data.TrainTestSplit(data, 0.2);

            testData = set.TestSet;
            trainData = set.TrainSet;
        }

        protected override ITransformer BuildAndTrainModel()
        {
            var estimator =
                context
                .Transforms.Text
                .FeaturizeText("Features", nameof(SentimentialData.Text))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression("Label", featureColumnName: "Features"));

            var model = estimator.Fit(trainData);

            return model;
        }

        protected override void Evaluate(ITransformer model)
        {
            var prediction = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(prediction, "Label");

            Console.WriteLine("Model quality");
            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"F1Score: {metrics.F1Score} => 1"); // мера баланса между точностью и полнотой
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve} => 1"); //уверенность модели в правильности классификации с положительными и отрицательными классами. Чем ближе к 1, тем лучше
        }
        protected override void UseModel(ITransformer model)
        {
            var predictionFunc = context.Model.CreatePredictionEngine<SentimentialData, SentimentialPrediction>(model);
            Console.WriteLine("Input data please");
            while (true)
            {
                var answer = Console.ReadLine();
                if (answer.ToLower() == "exit")
                {
                    break;
                }

                var example = new SentimentialData { Text = answer };
                var result = predictionFunc.Predict(example);

                Console.WriteLine($"Prediction: {result.Prediction}, Probability: {result.Probability}");
            }
        }
    }
}
